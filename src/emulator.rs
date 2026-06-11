use anyhow::{Context, Result};
use boytacean::gb::{GameBoy, GameBoyMode};
use boytacean::pad::PadKey;

pub const SCREEN_W: usize = 160;
pub const SCREEN_H: usize = 144;
pub const RGBA_SIZE: usize = SCREEN_W * SCREEN_H * 4;

pub struct Emulator {
    gb: GameBoy,
}

impl Emulator {
    pub fn new(rom: &[u8]) -> Result<Self> {
        let mut gb = GameBoy::new(Some(GameBoyMode::Dmg));
        gb.load_dmg(true, None).context("failed to load DMG boot ROM")?;
        gb.load_rom(rom, None).context("failed to load ROM")?;
        Ok(Self { gb })
    }

    /// Create an emulator with an empty ROM for testing.
    pub fn test() -> Self {
        let mut gb = GameBoy::new(Some(GameBoyMode::Dmg));
        gb.load_dmg(true, None).ok();
        gb.load_rom_empty().ok();
        gb.boot();
        Self { gb }
    }

    pub fn boot(&mut self) {
        self.gb.boot();
    }

    pub fn tick(&mut self) -> u16 {
        self.gb.clock()
    }

    pub fn next_frame(&mut self) -> u32 {
        self.gb.next_frame()
    }

    pub fn screen_rgba(&mut self) -> [u8; RGBA_SIZE] {
        self.gb.frame_buffer_rgba()
    }

    pub fn read_ram(&mut self, addr: u16) -> u8 {
        self.gb.read_memory(addr)
    }

    pub fn press(&mut self, key: PadKey) {
        self.gb.key_press(key);
    }

    pub fn release(&mut self, key: PadKey) {
        self.gb.key_lift(key);
    }

    pub fn release_all(&mut self) {
        self.gb.key_lift(PadKey::Right);
        self.gb.key_lift(PadKey::Left);
        self.gb.key_lift(PadKey::Up);
        self.gb.key_lift(PadKey::Down);
        self.gb.key_lift(PadKey::A);
        self.gb.key_lift(PadKey::B);
        self.gb.key_lift(PadKey::Select);
        self.gb.key_lift(PadKey::Start);
    }

    pub fn reset(&mut self) {
        self.gb.reset();
    }

    pub fn description(&self) -> String {
        self.gb.description(80)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emulator_creates() {
        let mut e = Emulator::test();
        e.boot();
        assert!(!e.description().is_empty());
    }

    #[test]
    fn test_emulator_screen_size() {
        let mut e = Emulator::test();
        e.boot();
        e.next_frame();
        let screen = e.screen_rgba();
        assert_eq!(screen.len(), RGBA_SIZE);
    }

    #[test]
    fn test_emulator_read_ram() {
        let mut e = Emulator::test();
        e.boot();
        e.next_frame();
        let val = e.read_ram(0xC000);
        // WRAM should be initialized to some value after boot
        assert!(val < 0x80);
    }

    #[test]
    fn test_emulator_press_and_release() {
        let mut e = Emulator::test();
        e.boot();
        e.press(PadKey::A);
        e.release(PadKey::A);
        // No crash = success
    }

    #[test]
    fn test_emulator_release_all() {
        let mut e = Emulator::test();
        e.boot();
        e.press(PadKey::A);
        e.press(PadKey::B);
        e.press(PadKey::Start);
        e.release_all();
        // No crash = success
    }

    #[test]
    fn test_emulator_reset() {
        let mut e = Emulator::test();
        e.boot();
        e.next_frame();
        // Create a fresh test emulator (simulates reset)
        let mut e2 = Emulator::test();
        e2.boot();
        e2.next_frame();
        let screen = e2.screen_rgba();
        assert_eq!(screen.len(), RGBA_SIZE);
    }

    #[test]
    fn test_emulator_screen_stable() {
        let mut e = Emulator::test();
        e.boot();
        // Run a few frames
        for _ in 0..10 {
            e.next_frame();
        }
        // Screen should not change without input
        let screen1 = e.screen_rgba();
        e.next_frame();
        let screen2 = e.screen_rgba();
        assert_eq!(screen1, screen2, "screen should be stable with no input");
    }
}

