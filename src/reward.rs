use crate::config::{GameConfig, ScannerState};
use crate::emulator::Emulator;

pub struct RewardSystem {
    prev_frame: Option<Vec<u8>>,
    scanner_state: ScannerState,
    scanners: Vec<crate::config::ScannerConfig>,
    last_breakdown: Vec<(String, f64)>,
    stale_steps: u32,
}

impl RewardSystem {
    pub fn new(config: Option<GameConfig>) -> Self {
        let scanners = config.map(|c| c.scanners).unwrap_or_default();
        Self {
            prev_frame: None,
            scanner_state: ScannerState::new(),
            scanners,
            last_breakdown: Vec::new(),
            stale_steps: 0,
        }
    }

    pub fn compute(&mut self, emulator: &mut Emulator, frame: &[u8]) -> f64 {
        let mut total = 0.0;
        self.last_breakdown.clear();

        if let Some(prev) = &self.prev_frame {
            if frame.len() == prev.len() {
                let mut diff = 0u64;
                for (a, b) in frame.iter().zip(prev.iter()) {
                    diff += (*a as i16 - *b as i16).unsigned_abs() as u64;
                }
                let total_pixels = (frame.len() / 4) as f64;
                let avg_diff = diff as f64 / total_pixels;

                if avg_diff > 5.0 {
                    let novelty = (avg_diff / 255.0).min(1.0) * 0.5;
                    total += novelty;
                    self.last_breakdown.push(("screen_novelty".into(), novelty));
                    self.stale_steps = 0;
                } else {
                    self.stale_steps += 1;
                    if self.stale_steps > 60 {
                        let penalty = -0.05 * (self.stale_steps as f64 / 60.0).min(10.0);
                        total += penalty;
                        self.last_breakdown.push(("stale_penalty".into(), penalty));
                    }
                }
            }
        }

        if !self.scanners.is_empty() {
            let scanner_reward = self.scanner_state.compute_reward(
                &self.scanners,
                |addr| emulator.read_ram(addr),
            );
            if scanner_reward != 0.0 {
                total += scanner_reward;
                self.last_breakdown.push(("scanner".into(), scanner_reward));
            }
        }

        let result = total.clamp(-10.0, 10.0);

        self.prev_frame = Some(frame.to_vec());

        result
    }

    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.scanner_state = ScannerState::new();
        self.last_breakdown.clear();
        self.stale_steps = 0;
    }

    pub fn last_breakdown(&self) -> &[(String, f64)] {
        &self.last_breakdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScannerConfig;

    fn dummy_frame(value: u8) -> Vec<u8> {
        vec![value; 160 * 144 * 4]
    }

    fn dummy_emulator() -> Emulator {
        Emulator::test()
    }

    #[test]
    fn test_reward_screen_novelty_positive() {
        let mut rs = RewardSystem::new(None);
        let mut emu = dummy_emulator();

        // First frame establishes baseline
        let r1 = rs.compute(&mut emu, &dummy_frame(0));
        assert_eq!(r1, 0.0);

        // Different frame gives novelty
        let r2 = rs.compute(&mut emu, &dummy_frame(255));
        assert!(r2 > 0.0, "expected positive novelty, got {r2}");
    }

    #[test]
    fn test_reward_screen_stale_penalty() {
        let mut rs = RewardSystem::new(None);
        let mut emu = dummy_emulator();
        let frame = dummy_frame(128);

        rs.compute(&mut emu, &frame); // baseline

        // 60 stale frames = no penalty yet
        for _ in 0..60 {
            rs.compute(&mut emu, &frame);
        }

        // 61st should have a small penalty
        let r = rs.compute(&mut emu, &frame);
        assert!(r < 0.0, "expected penalty, got {r}");
    }

    #[test]
    fn test_reward_reset_clears() {
        let mut rs = RewardSystem::new(None);
        let mut emu = dummy_emulator();

        rs.compute(&mut emu, &dummy_frame(0));
        let r1 = rs.compute(&mut emu, &dummy_frame(255));
        assert!(r1 > 0.0);

        rs.reset();
        let r2 = rs.compute(&mut emu, &dummy_frame(0));
        assert_eq!(r2, 0.0, "reset should clear prev_frame");
    }

    #[test]
    fn test_reward_breakdown_populated() {
        let mut rs = RewardSystem::new(None);
        let mut emu = dummy_emulator();
        rs.compute(&mut emu, &dummy_frame(0));
        let _ = rs.compute(&mut emu, &dummy_frame(255));
        let breakdown = rs.last_breakdown();
        assert!(!breakdown.is_empty());
        assert_eq!(breakdown[0].0, "screen_novelty");
    }
}
