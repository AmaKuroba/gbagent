use anyhow::{Context, Result};
use boytacean::gb::{GameBoy, GameBoyMode};
use boytacean::pad::PadKey;

pub const SCREEN_W: usize = 160;
pub const SCREEN_H: usize = 144;
pub const RGBA_SIZE: usize = SCREEN_W * SCREEN_H * 4;

/// Wrapper around `boytacean::gb::GameBoy` for headless use.
pub struct Emulator {
    gb: GameBoy,
}

impl Emulator {
    /// Create a new DMG Game Boy with the given ROM data.
    pub fn new(rom: &[u8]) -> Result<Self> {
        let mut gb = GameBoy::new(Some(GameBoyMode::Dmg));
        gb.load_dmg(true, None).context("failed to load DMG boot ROM")?;
        gb.load_rom(rom, None).context("failed to load ROM")?;
        Ok(Self { gb })
    }

    /// Skip boot sequence and jump straight to cartridge execution.
    pub fn boot(&mut self) {
        self.gb.boot();
    }

    /// Advance emulation by one CPU instruction/tick.
    /// Returns the number of CPU cycles executed.
    pub fn tick(&mut self) -> u16 {
        self.gb.clock()
    }

    /// Advance emulation until the next frame is complete.
    /// Returns the number of cycles clocked.
    pub fn next_frame(&mut self) -> u32 {
        self.gb.next_frame()
    }

    /// Return the current RGBA framebuffer (160×144×4).
    pub fn screen_rgba(&mut self) -> [u8; RGBA_SIZE] {
        self.gb.frame_buffer_rgba()
    }

    /// Read a byte from Game Boy memory.
    pub fn read_ram(&mut self, addr: u16) -> u8 {
        self.gb.read_memory(addr)
    }

    /// Press a Game Boy button.
    pub fn press(&mut self, key: PadKey) {
        self.gb.key_press(key);
    }

    /// Release a Game Boy button.
    pub fn release(&mut self, key: PadKey) {
        self.gb.key_lift(key);
    }

    /// Release all buttons.
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

    /// Reset the entire system.
    pub fn reset(&mut self) {
        self.gb.reset();
    }

    /// Return a human-readable description of the loaded system.
    pub fn description(&self) -> String {
        self.gb.description(80)
    }
}
