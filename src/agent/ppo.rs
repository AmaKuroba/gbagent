use candle_core::{Device, Tensor, Module};
use candle_nn::{lstm, linear, AdamW, Linear, LSTMConfig, Optimizer, RNN, VarBuilder, VarMap};

use super::encoder::{ViTConfig, ViTEncoder};

pub const DPAD_SIZE: usize = 5;
pub const BTN_SIZE: usize = 4;
pub const EMBED_DIM: usize = 384;
pub const LSTM_HIDDEN: usize = 256;

#[derive(Debug, Clone)]
pub struct PPOConfig {
    pub learning_rate: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub clip_epsilon: f64,
    pub entropy_coef: f64,
    pub value_coef: f64,
    pub max_grad_norm: f64,
    pub rollout_len: usize,
    pub batch_size: usize,
    pub ppo_epochs: usize,
    pub total_steps: usize,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_grad_norm: 0.5,
            rollout_len: 128,
            batch_size: 32,
            ppo_epochs: 4,
            total_steps: 6_000_000,
        }
    }
}

/// Actor-Critic: ViT → LSTM → policy/value heads
pub struct ActorCritic {
    pub encoder: ViTEncoder,
    pub lstm: candle_nn::LSTM,
    pub dpad_head: Linear,
    pub btn_head: Linear,
    pub value_head: Linear,
    pub var_map: VarMap,
    pub device: Device,
    pub hidden: Option<<candle_nn::LSTM as RNN>::State>,
}

impl ActorCritic {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let vit_cfg = ViTConfig::default();
        let encoder = ViTEncoder::new(&vit_cfg, &device)?;

        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, device);

        let lstm = lstm(EMBED_DIM, LSTM_HIDDEN, LSTMConfig::default(), vb.pp("lstm"))?;
        let dpad_head = linear(LSTM_HIDDEN, DPAD_SIZE, vb.pp("dpad_head"))?;
        let btn_head = linear(LSTM_HIDDEN, BTN_SIZE, vb.pp("btn_head"))?;
        let value_head = linear(LSTM_HIDDEN, 1, vb.pp("value_head"))?;

        Ok(Self {
            encoder, lstm, dpad_head, btn_head, value_head,
            var_map, device: device.clone(), hidden: None,
        })
    }

    /// Forward pass. Returns (dpad_logits, btn_logits, value).
    pub fn forward(&mut self, obs: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let encoded = self.encoder.forward(obs)?;
        let state = self.hidden.clone()
            .unwrap_or_else(|| self.lstm.zero_state(1).unwrap());
        let new_state = self.lstm.step(&encoded, &state)?;
        let lstm_out = new_state.h.clone();
        self.hidden = Some(new_state);

        let dpad_logits = self.dpad_head.forward(&lstm_out)?;
        let btn_logits = self.btn_head.forward(&lstm_out)?;
        let value = self.value_head.forward(&lstm_out)?;

        Ok((dpad_logits, btn_logits, value))
    }

    pub fn reset_hidden(&mut self) {
        self.hidden = None;
    }

    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub obs: Vec<u8>,
    pub dpad_action: usize,
    pub btn_action: usize,
    pub dpad_log_prob: f64,
    pub btn_log_prob: f64,
    pub reward: f64,
    pub done: bool,
    pub value: f64,
}

pub struct RolloutBuffer {
    pub transitions: Vec<Transition>,
    pub capacity: usize,
}

impl RolloutBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { transitions: Vec::with_capacity(capacity), capacity }
    }
    pub fn push(&mut self, t: Transition) { self.transitions.push(t); }
    pub fn clear(&mut self) { self.transitions.clear(); }
    pub fn len(&self) -> usize { self.transitions.len() }
}

pub struct PPOAgent {
    pub network: ActorCritic,
    pub config: PPOConfig,
    pub buffer: RolloutBuffer,
    pub step_count: usize,
}

impl PPOAgent {
    pub fn new(config: PPOConfig, device: &Device) -> anyhow::Result<Self> {
        let network = ActorCritic::new(device)?;
        let buffer = RolloutBuffer::new(config.rollout_len);
        Ok(Self { network, config, buffer, step_count: 0 })
    }

    pub fn act(
        &mut self, obs: &[u8], training: bool,
    ) -> anyhow::Result<(usize, usize, f64, f64, f64)> {
        let obs_t = Tensor::from_slice(obs, (1, 4, 144, 160), &self.network.device)?
            .to_dtype(candle_core::DType::F32)?;
        let (dpad_logits, btn_logits, value) = self.network.forward(&obs_t)?;
        let value_v = value.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()? as f64;

        let dpad_probs = candle_nn::ops::softmax(&dpad_logits, 1)?;
        let btn_probs = candle_nn::ops::softmax(&btn_logits, 1)?;

        let dpad_action = if training {
            sample_categorical(&dpad_probs)?
        } else {
            dpad_probs.argmax(1)?.squeeze(1)?.squeeze(0)?.to_scalar::<u32>()? as usize
        };
        let btn_action = if training {
            sample_categorical(&btn_probs)?
        } else {
            btn_probs.argmax(1)?.squeeze(1)?.squeeze(0)?.to_scalar::<u32>()? as usize
        };

        let dpad_lp = dpad_probs.narrow(1, dpad_action, 1)?.squeeze(1)?
            .log()?.squeeze(0)?.to_scalar::<f32>()? as f64;
        let btn_lp = btn_probs.narrow(1, btn_action, 1)?.squeeze(1)?
            .log()?.squeeze(0)?.to_scalar::<f32>()? as f64;

        Ok((dpad_action, btn_action, dpad_lp, btn_lp, value_v))
    }

    pub fn update(&mut self) -> anyhow::Result<AgentStats> {
        let n = self.buffer.len();
        if n < 2 { return Ok(AgentStats::default()); }
        let cfg = &self.config;

        let mut advantages = vec![0.0f64; n];
        let mut returns = vec![0.0f64; n];
        let mut gae = 0.0;
        for t in (0..n - 1).rev() {
            let next_value = self.buffer.transitions[t + 1].value;
            let not_done = if self.buffer.transitions[t + 1].done { 0.0 } else { 1.0 };
            let delta = self.buffer.transitions[t].reward
                + cfg.gamma * next_value * not_done
                - self.buffer.transitions[t].value;
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae;
            advantages[t] = gae;
            returns[t] = advantages[t] + self.buffer.transitions[t].value;
        }

        let mean = advantages.iter().sum::<f64>() / n as f64;
        let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-8);
        for a in advantages.iter_mut() { *a = (*a - mean) / std; }

        let (obs, dpad_acts, btn_acts, old_dpad_lp, old_btn_lp, advs, rets) = {
            let t = &self.buffer.transitions;
            let obs: Vec<Vec<u8>> = t.iter().map(|x| x.obs.clone()).collect();
            let dpad_acts: Vec<u32> = t.iter().map(|x| x.dpad_action as u32).collect();
            let btn_acts: Vec<u32> = t.iter().map(|x| x.btn_action as u32).collect();
            let old_dpad_lp: Vec<f32> = t.iter().map(|x| x.dpad_log_prob as f32).collect();
            let old_btn_lp: Vec<f32> = t.iter().map(|x| x.btn_log_prob as f32).collect();
            let advs: Vec<f32> = advantages.iter().map(|x| *x as f32).collect();
            let rets: Vec<f32> = returns.iter().map(|x| *x as f32).collect();
            (obs, dpad_acts, btn_acts, old_dpad_lp, old_btn_lp, advs, rets)
        };

        let mut opt = AdamW::new(
            self.network.var_map().all_vars(),
            candle_nn::ParamsAdamW { lr: cfg.learning_rate, ..Default::default() },
        )?;

        let device = self.network.device.clone();
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut update_count = 0;

        for _epoch in 0..cfg.ppo_epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() { let j = fastrand::usize(..i + 1); indices.swap(i, j); }

            for chunk in indices.chunks(cfg.batch_size) {
                let batch = chunk.to_vec();
                let bs = batch.len();
                let mut batch_obs = Vec::with_capacity(bs * 4 * 144 * 160);
                for &idx in &batch { batch_obs.extend_from_slice(&obs[idx]); }
                let obs_t = Tensor::from_slice(&batch_obs, (bs, 4, 144, 160), &device)?
                    .to_dtype(candle_core::DType::F32)?;

                self.network.reset_hidden();
                let (dpad_logits, btn_logits, values) = self.network.forward(&obs_t)?;
                let dpad_probs = candle_nn::ops::softmax(&dpad_logits, 1)?;
                let btn_probs = candle_nn::ops::softmax(&btn_logits, 1)?;

                let dpad_acts_t = Tensor::from_slice(
                    &batch.iter().map(|&i| dpad_acts[i]).collect::<Vec<u32>>(), bs, &device)?;
                let btn_acts_t = Tensor::from_slice(
                    &batch.iter().map(|&i| btn_acts[i]).collect::<Vec<u32>>(), bs, &device)?;

                let new_dpad_lp = dpad_probs.gather(&dpad_acts_t.unsqueeze(1)?, 1)?
                    .squeeze(1)?.log()?;
                let new_btn_lp = btn_probs.gather(&btn_acts_t.unsqueeze(1)?, 1)?
                    .squeeze(1)?.log()?;

                let old_dpad_lp_t = Tensor::from_slice(
                    &batch.iter().map(|&i| old_dpad_lp[i]).collect::<Vec<f32>>(), bs, &device)?;
                let old_btn_lp_t = Tensor::from_slice(
                    &batch.iter().map(|&i| old_btn_lp[i]).collect::<Vec<f32>>(), bs, &device)?;

                // ratio = exp(new_log_prob - old_log_prob)
                let ratio_dpad = ((new_dpad_lp - old_dpad_lp_t)?).exp()?;
                let ratio_btn = ((new_btn_lp - old_btn_lp_t)?).exp()?;
                let ratio = ((ratio_dpad + ratio_btn)? / 2.0)?;

                let adv_t = Tensor::from_slice(
                    &batch.iter().map(|&i| advs[i]).collect::<Vec<f32>>(), bs, &device)?;

                // surr1 = ratio * adv
                let surr1 = ratio.mul(&adv_t)?;
                let clipped = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)?;
                let surr2 = clipped.mul(&adv_t)?;
                let policy_loss = surr1.minimum(&surr2)?.sum(0)?.squeeze(0)?.neg()?;

                let ret_t = Tensor::from_slice(
                    &batch.iter().map(|&i| rets[i]).collect::<Vec<f32>>(), bs, &device)?;
                let value_loss = ((values.squeeze(1)? - ret_t)?).sqr()?.mean(0)?;

                // Entropy: -sum(p * log(p))
                let dpad_entropy = ((dpad_probs.neg()? * dpad_probs.log()?)?).sum(1)?.mean(0)?;
                let btn_entropy = ((btn_probs.neg()? * btn_probs.log()?)?).sum(1)?.mean(0)?;

                let pl_v = policy_loss.to_scalar::<f32>()? as f64;
                let vl_v = value_loss.to_scalar::<f32>()? as f64;
                let de_v = dpad_entropy.to_scalar::<f32>()? as f64;
                let be_v = btn_entropy.to_scalar::<f32>()? as f64;
                let entropy_v = (de_v + be_v) / 2.0;

                let loss_val = pl_v + cfg.value_coef * vl_v - cfg.entropy_coef * entropy_v;
                let loss = Tensor::new(loss_val, &device)?;
                opt.backward_step(&loss)?;

                total_policy_loss += pl_v;
                total_value_loss += vl_v;
                total_entropy += entropy_v;
                update_count += 1;
            }
        }

        self.step_count += n;
        self.buffer.clear();

        Ok(AgentStats {
            policy_loss: total_policy_loss / update_count.max(1) as f64,
            value_loss: total_value_loss / update_count.max(1) as f64,
            entropy_loss: total_entropy / update_count.max(1) as f64,
        })
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.network.var_map().save(path)?;
        log::info!("model saved to {path}");
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> anyhow::Result<()> {
        self.network.var_map.load(path)?;
        log::info!("model loaded from {path}");
        Ok(())
    }
}

fn sample_categorical(probs: &Tensor) -> anyhow::Result<usize> {
    let uniform = Tensor::rand(0.0f32, 1.0f32, probs.dims(), probs.device())?;
    let gumbel = uniform.log()?.neg()?.log()?;
    let logits = probs.log()?;
    let sampled = (logits + gumbel)?.argmax(1)?.squeeze(1)?.squeeze(0)?;
    Ok(sampled.to_scalar::<u32>()? as usize)
}

#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy_loss: f64,
}
