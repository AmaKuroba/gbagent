use std::collections::HashMap;

use serde::Deserialize;

/// Scanner configuration for a single RAM-based reward signal.
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

/// Game-specific configuration loaded from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct GameConfig {
    pub game: String,
    pub scanners: Vec<ScannerConfig>,
}

/// Load a game config from a YAML file path.
pub fn load_game_config(path: &str) -> anyhow::Result<GameConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: GameConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}

/// ScannerState tracks the previous values for delta-type RAM scanners.
#[derive(Debug, Default)]
pub struct ScannerState {
    prev_values: HashMap<u16, Vec<u8>>,
}

impl ScannerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute reward from all configured scanners given a read function.
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
                        // delta: reward_per_unit per level difference
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
                        // delta_multi: sum of absolute differences per byte
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
