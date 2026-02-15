use rand::Rng;

pub const INPUT_SIZE: usize = 22;
const HIDDEN1: usize = 256;
const HIDDEN2: usize = 64;
pub const OUTPUT_SIZE: usize = 4;

struct DenseLayer {
    weights: Vec<f32>, // [in_size × out_size], row-major: w[i * out + j]
    biases: Vec<f32>,
    relu: bool,
    in_size: usize,
    out_size: usize,
    m_w: Vec<f32>,
    v_w: Vec<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

impl DenseLayer {
    fn new(in_size: usize, out_size: usize, relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_size + out_size) as f32).sqrt();
        let n = in_size * out_size;
        let weights: Vec<f32> = (0..n)
            .map(|_| rng.gen::<f32>() * 2.0 * limit - limit)
            .collect();

        DenseLayer {
            weights,
            biases: vec![0.0; out_size],
            relu,
            in_size,
            out_size,
            m_w: vec![0.0; n],
            v_w: vec![0.0; n],
            m_b: vec![0.0; out_size],
            v_b: vec![0.0; out_size],
        }
    }

    fn forward_single(&self, input: &[f32], output: &mut [f32]) {
        for j in 0..self.out_size {
            let mut sum = self.biases[j];
            for i in 0..self.in_size {
                sum += input[i] * self.weights[i * self.out_size + j];
            }
            output[j] = if self.relu { sum.max(0.0) } else { sum };
        }
    }

    /// Returns (z, a) flattened as [bs * out_size]
    fn forward_batch(&self, input: &[f32], bs: usize) -> (Vec<f32>, Vec<f32>) {
        let mut z = vec![0.0f32; bs * self.out_size];
        let mut a = vec![0.0f32; bs * self.out_size];

        for b in 0..bs {
            let inp = &input[b * self.in_size..(b + 1) * self.in_size];
            for j in 0..self.out_size {
                let mut sum = self.biases[j];
                for i in 0..self.in_size {
                    sum += inp[i] * self.weights[i * self.out_size + j];
                }
                let idx = b * self.out_size + j;
                z[idx] = sum;
                a[idx] = if self.relu { sum.max(0.0) } else { sum };
            }
        }

        (z, a)
    }

    fn adam_update(&mut self, gw: &[f32], gb: &[f32], lr: f32, t: usize) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);

        for i in 0..self.weights.len() {
            self.m_w[i] = b1 * self.m_w[i] + (1.0 - b1) * gw[i];
            self.v_w[i] = b2 * self.v_w[i] + (1.0 - b2) * gw[i] * gw[i];
            let mh = self.m_w[i] / bc1;
            let vh = self.v_w[i] / bc2;
            self.weights[i] -= lr * mh / (vh.sqrt() + eps);
        }

        for i in 0..self.biases.len() {
            self.m_b[i] = b1 * self.m_b[i] + (1.0 - b1) * gb[i];
            self.v_b[i] = b2 * self.v_b[i] + (1.0 - b2) * gb[i] * gb[i];
            let mh = self.m_b[i] / bc1;
            let vh = self.v_b[i] / bc2;
            self.biases[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

pub struct Network {
    layers: Vec<DenseLayer>,
    t: usize,
}

impl Network {
    pub fn new() -> Self {
        Network {
            layers: vec![
                DenseLayer::new(INPUT_SIZE, HIDDEN1, true),
                DenseLayer::new(HIDDEN1, HIDDEN2, true),
                DenseLayer::new(HIDDEN2, OUTPUT_SIZE, false),
            ],
            t: 0,
        }
    }

    pub fn forward(&self, input: &[f32]) -> [f32; OUTPUT_SIZE] {
        let mut buf1 = vec![0.0f32; HIDDEN1];
        let mut buf2 = vec![0.0f32; HIDDEN2];
        let mut out = [0.0f32; OUTPUT_SIZE];

        self.layers[0].forward_single(input, &mut buf1);
        self.layers[1].forward_single(&buf1, &mut buf2);
        self.layers[2].forward_single(&buf2, &mut out);

        out
    }

    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Vec<[f32; OUTPUT_SIZE]> {
        inputs.iter().map(|inp| self.forward(inp)).collect()
    }

    pub fn train_batch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], lr: f32) {
        self.t += 1;
        let bs = inputs.len();
        let bsf = bs as f32;

        // Flatten inputs: [bs * INPUT_SIZE]
        let flat_in: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();

        // Forward all layers, cache z (pre-activation) and a (post-activation)
        let (z0, a0) = self.layers[0].forward_batch(&flat_in, bs);
        let (z1, a1) = self.layers[1].forward_batch(&a0, bs);
        let (_z2, a2) = self.layers[2].forward_batch(&a1, bs);

        // --- Backprop ---

        // dL/dz2 = (a2 - target) * 2/output_size  (layer 2 is linear, so dL/dz = dL/da)
        let mut dz = vec![0.0f32; bs * OUTPUT_SIZE];
        for b in 0..bs {
            for j in 0..OUTPUT_SIZE {
                let idx = b * OUTPUT_SIZE + j;
                dz[idx] = (a2[idx] - targets[b][j]) * (2.0 / OUTPUT_SIZE as f32);
            }
        }

        // Layer 2: gw2 = a1^T @ dz / bs, gb2 = sum(dz) / bs, delta = dz @ W2^T
        let l = &self.layers[2];
        let gw2 = matmul_at_b(&a1, &dz, l.in_size, l.out_size, bs, bsf);
        let gb2 = sum_cols(&dz, l.out_size, bs, bsf);
        let mut delta = matmul_a_bt(&dz, &l.weights, l.out_size, l.in_size, bs);

        // Apply relu'(z1) to get dL/dz1
        for i in 0..delta.len() {
            if z1[i] <= 0.0 {
                delta[i] = 0.0;
            }
        }
        let dz1 = delta;

        // Layer 1: gw1 = a0^T @ dz1 / bs, gb1 = sum(dz1) / bs, delta = dz1 @ W1^T
        let l = &self.layers[1];
        let gw1 = matmul_at_b(&a0, &dz1, l.in_size, l.out_size, bs, bsf);
        let gb1 = sum_cols(&dz1, l.out_size, bs, bsf);
        let mut delta = matmul_a_bt(&dz1, &l.weights, l.out_size, l.in_size, bs);

        // Apply relu'(z0) to get dL/dz0
        for i in 0..delta.len() {
            if z0[i] <= 0.0 {
                delta[i] = 0.0;
            }
        }
        let dz0 = delta;

        // Layer 0: gw0 = input^T @ dz0 / bs, gb0 = sum(dz0) / bs
        let l = &self.layers[0];
        let gw0 = matmul_at_b(&flat_in, &dz0, l.in_size, l.out_size, bs, bsf);
        let gb0 = sum_cols(&dz0, l.out_size, bs, bsf);

        // Adam updates
        self.layers[2].adam_update(&gw2, &gb2, lr, self.t);
        self.layers[1].adam_update(&gw1, &gb1, lr, self.t);
        self.layers[0].adam_update(&gw0, &gb0, lr, self.t);
    }

    pub fn clone_weights(&self) -> Self {
        Network {
            layers: self
                .layers
                .iter()
                .map(|l| DenseLayer {
                    weights: l.weights.clone(),
                    biases: l.biases.clone(),
                    relu: l.relu,
                    in_size: l.in_size,
                    out_size: l.out_size,
                    m_w: vec![0.0; l.m_w.len()],
                    v_w: vec![0.0; l.v_w.len()],
                    m_b: vec![0.0; l.m_b.len()],
                    v_b: vec![0.0; l.v_b.len()],
                })
                .collect(),
            t: 0,
        }
    }

    /// Returns (weights, biases, in_size, out_size) for export
    pub fn layer_info(&self, idx: usize) -> (&[f32], &[f32], usize, usize) {
        let l = &self.layers[idx];
        (&l.weights, &l.biases, l.in_size, l.out_size)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Compute A^T @ B / scale, where A is [bs × m] and B is [bs × n], result is [m × n]
fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, bs: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for s in 0..bs {
        let a_row = &a[s * m..(s + 1) * m];
        let b_row = &b[s * n..(s + 1) * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] += a_row[i] * b_row[j];
            }
        }
    }
    for v in out.iter_mut() {
        *v /= scale;
    }
    out
}

/// Compute A @ B^T, where A is [bs × n] and B is [m × n] (stored row-major), result is [bs × m]
/// B here is the weight matrix W[in × out] = [m × n_cols_of_W]
/// Wait - W is [in_size × out_size]. delta is [bs × out_size].
/// We want delta @ W^T = [bs × out_size] @ [out_size × in_size] = [bs × in_size]
/// W stored as [in_size × out_size] row-major, so W^T[j, i] = W[i * out + j]
fn matmul_a_bt(a: &[f32], w: &[f32], out_size: usize, in_size: usize, bs: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; bs * in_size];
    for b in 0..bs {
        let a_row = &a[b * out_size..(b + 1) * out_size];
        for i in 0..in_size {
            let mut sum = 0.0f32;
            for j in 0..out_size {
                // W[i, j] = w[i * out_size + j]
                sum += a_row[j] * w[i * out_size + j];
            }
            result[b * in_size + i] = sum;
        }
    }
    result
}

/// Sum columns across batch: result[j] = sum_b(data[b * cols + j]) / scale
fn sum_cols(data: &[f32], cols: usize, bs: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for b in 0..bs {
        for j in 0..cols {
            out[j] += data[b * cols + j];
        }
    }
    for v in out.iter_mut() {
        *v /= scale;
    }
    out
}
