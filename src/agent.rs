use crate::nn::Network;
use rand::Rng;
use std::collections::VecDeque;

pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct DQNAgent {
    pub network: Network,
    pub target_network: Network,
    replay_buffer: VecDeque<Experience>,
    buffer_size: usize,
    batch_size: usize,
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub learning_rate: f32,
    target_update_every: u64,
    train_every: u64,
    step_count: u64,
    episodes_since_target_update: u64,
}

impl DQNAgent {
    pub fn new() -> Self {
        let network = Network::new();
        let target_network = network.clone_weights();
        DQNAgent {
            network,
            target_network,
            replay_buffer: VecDeque::with_capacity(50_000),
            buffer_size: 50_000,
            batch_size: 64,
            gamma: 0.95,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
            learning_rate: 0.001,
            target_update_every: 10,
            train_every: 4,
            step_count: 0,
            episodes_since_target_update: 0,
        }
    }

    pub fn act(&self, features: &[f32]) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.epsilon {
            rng.gen_range(0..4)
        } else {
            self.act_greedy(features)
        }
    }

    pub fn act_greedy(&self, features: &[f32]) -> usize {
        let q = self.network.forward(features);
        q.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    pub fn remember(&mut self, exp: Experience) {
        if self.replay_buffer.len() >= self.buffer_size {
            self.replay_buffer.pop_front();
        }
        self.replay_buffer.push_back(exp);
    }

    pub fn buffer_len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn step_and_train(&mut self) {
        self.step_count += 1;
        if self.step_count % self.train_every != 0 {
            return;
        }
        self.train();
    }

    fn train(&mut self) {
        if self.replay_buffer.len() < self.batch_size {
            return;
        }

        let mut rng = rand::thread_rng();
        let buf_len = self.replay_buffer.len();

        // Uniform random sampling
        let indices: Vec<usize> = (0..self.batch_size)
            .map(|_| rng.gen_range(0..buf_len))
            .collect();

        let states: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| self.replay_buffer[i].state.clone())
            .collect();
        let next_states: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| self.replay_buffer[i].next_state.clone())
            .collect();

        let current_qs = self.network.predict_batch(&states);
        let next_qs = self.target_network.predict_batch(&next_states);

        let mut targets: Vec<Vec<f32>> = current_qs.iter().map(|q| q.to_vec()).collect();

        for (idx, &buf_idx) in indices.iter().enumerate() {
            let exp = &self.replay_buffer[buf_idx];
            targets[idx][exp.action] = if exp.done {
                exp.reward
            } else {
                let max_next = next_qs[idx]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                exp.reward + self.gamma * max_next
            };
        }

        self.network
            .train_batch(&states, &targets, self.learning_rate);
    }

    pub fn end_episode(&mut self) {
        self.decay_epsilon();
        self.episodes_since_target_update += 1;
        if self.episodes_since_target_update >= self.target_update_every {
            self.target_network = self.network.clone_weights();
            self.episodes_since_target_update = 0;
        }
    }

    fn decay_epsilon(&mut self) {
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }
}
