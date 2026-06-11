use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ScannerConfig {
    pub name: String,
    pub ram_addr: u16,
    #[serde(rename = "type")]
    pub scan_type: String,
    #[serde(default)]
    pub reward_on_change: f64,
    #[serde(default)]
    pub reward_per_unit: f64,
    #[serde(default = "one")]
    pub length: u16,
}

fn one() -> u16 { 1 }

#[derive(Debug, Clone, Deserialize)]
pub struct GameConfig {
    pub game: String,
    pub scanners: Vec<ScannerConfig>,
}

pub fn load_game_config(path: &str) -> anyhow::Result<GameConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: GameConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}

#[derive(Debug, Default)]
pub struct ScannerState {
    prev_values: HashMap<u16, Vec<u8>>,
}

impl ScannerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_reward(
        &mut self,
        scanners: &[ScannerConfig],
        mut read_byte: impl FnMut(u16) -> u8,
    ) -> f64 {
        let mut total = 0.0;
        for scanner in scanners {
            match scanner.scan_type.as_str() {
                "discrete" | "delta" => {
                    let addr = scanner.ram_addr;
                    let curr = read_byte(addr);
                    let prev = self.prev_values.entry(addr).or_insert_with(|| vec![curr]);
                    let old_val = prev[0];
                    if scanner.scan_type == "discrete" {
                        if curr != old_val {
                            total += scanner.reward_on_change;
                        }
                    } else {
                        let diff = (curr as i16 - old_val as i16).unsigned_abs() as f64;
                        total += diff * scanner.reward_per_unit;
                    }
                    prev[0] = curr;
                }
                "multi_byte" | "delta_multi" => {
                    let len = scanner.length.max(1) as usize;
                    let mut curr_bytes = Vec::with_capacity(len);
                    for i in 0..len {
                        curr_bytes.push(read_byte(scanner.ram_addr + i as u16));
                    }
                    let prev = self.prev_values
                        .entry(scanner.ram_addr)
                        .or_insert_with(|| vec![0u8; len]);
                    if scanner.scan_type == "multi_byte" {
                        if curr_bytes != *prev {
                            total += scanner.reward_on_change;
                        }
                    } else {
                        let diff: f64 = curr_bytes.iter().zip(prev.iter())
                            .map(|(c, p)| (*c as i16 - *p as i16).unsigned_abs() as f64)
                            .sum();
                        total += diff * scanner.reward_per_unit;
                    }
                    *prev = curr_bytes;
                }
                _ => {}
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scanner(name: &str, ty: &str, addr: u16, reward: f64) -> ScannerConfig {
        ScannerConfig {
            name: name.into(),
            ram_addr: addr,
            scan_type: ty.into(),
            reward_on_change: reward,
            reward_per_unit: reward,
            length: 1,
        }
    }

    #[test]
    fn test_scanner_discrete_no_change() {
        let mut state = ScannerState::new();
        let scanners = vec![make_scanner("test", "discrete", 0xC000, 1.0)];
        let mut mem = |_: u16| 0x42u8;
        let reward = state.compute_reward(&scanners, &mut mem);
        assert_eq!(reward, 0.0, "no change on first read");
        // second read, same value
        let reward = state.compute_reward(&scanners, &mut mem);
        assert_eq!(reward, 0.0, "no change when same");
    }

    #[test]
    fn test_scanner_discrete_on_change() {
        let mut state = ScannerState::new();
        let scanners = vec![make_scanner("test", "discrete", 0xC000, 1.0)];
        let mut mem = |a: u16| -> u8 {
            if a == 0xC000 { 0x42 } else { 0 }
        };
        state.compute_reward(&scanners, &mut mem);
        let mut mem = |a: u16| -> u8 {
            if a == 0xC000 { 0xFF } else { 0 }
        };
        let reward = state.compute_reward(&scanners, &mut mem);
        assert!((reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scanner_delta() {
        let mut state = ScannerState::new();
        let scanners = vec![make_scanner("test", "delta", 0xC000, 0.5)];
        let mut mem = |a: u16| -> u8 {
            if a == 0xC000 { 10 } else { 0 }
        };
        state.compute_reward(&scanners, &mut mem);
        let mut mem = |a: u16| -> u8 {
            if a == 0xC000 { 15 } else { 0 }
        };
        let reward = state.compute_reward(&scanners, &mut mem);
        assert!((reward - 2.5).abs() < 1e-6, "5 diff * 0.5 per unit = 2.5, got {reward}");
    }

    #[test]
    fn test_scanner_multi_byte_change() {
        let mut state = ScannerState::new();
        let scanners = vec![ScannerConfig {
            name: "test".into(),
            ram_addr: 0xC000,
            scan_type: "multi_byte".into(),
            reward_on_change: 2.0,
            reward_per_unit: 0.0,
            length: 4,
        }];
        let mut mem = |a: u16| -> u8 { 0 };
        state.compute_reward(&scanners, &mut mem);
        let mut mem = |a: u16| -> u8 {
            if a == 0xC002 { 1 } else { 0 }
        };
        let reward = state.compute_reward(&scanners, &mut mem);
        assert!((reward - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_game_config_invalid_path() {
        let result = load_game_config("/nonexistent/file.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_scanner_invalid_type_ignored() {
        let mut state = ScannerState::new();
        let scanners = vec![make_scanner("bad", "unknown", 0xC000, 1.0)];
        let mut mem = |_: u16| 0u8;
        let reward = state.compute_reward(&scanners, &mut mem);
        assert_eq!(reward, 0.0);
    }
}
