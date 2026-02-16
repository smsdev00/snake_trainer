mod agent;
mod engine;
mod features;
mod nn;

use agent::{DQNAgent, Experience};
use engine::SnakeEngine;
use features::extract_features;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

const GRID_SIZE: i32 = 20;

struct Config {
    episodes: u64,
    print_every: u64,
    save_every: u64,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut map: HashMap<String, String> = HashMap::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: rust_entrenador [OPTIONS]");
            println!();
            println!("Options:");
            println!("  --episodes <N>     Number of training episodes  [default: 100000]");
            println!("  --print-every <N>  Print stats every N episodes [default: 100]");
            println!("  --save-every <N>   Save model every N episodes  [default: 5000]");
            println!("  -h, --help         Show this help");
            std::process::exit(0);
        }
        if args[i].starts_with("--") && i + 1 < args.len() {
            map.insert(args[i].clone(), args[i + 1].clone());
            i += 2;
        } else {
            eprintln!("Unknown argument: {}", args[i]);
            std::process::exit(1);
        }
    }

    Config {
        episodes: map
            .get("--episodes")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000),
        print_every: map
            .get("--print-every")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100),
        save_every: map
            .get("--save-every")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5_000),
    }
}

fn export_model(agent: &DQNAgent, filename: &str) {
    let mut weight_bytes: Vec<u8> = Vec::new();
    let mut weight_specs: Vec<serde_json::Value> = Vec::new();

    let layer_names = ["dense", "dense_1", "dense_2"];

    for i in 0..agent.network.num_layers() {
        let (weights, biases, in_size, out_size) = agent.network.layer_info(i);

        // Weights: stored as [in_size × out_size] row-major, TF.js expects same layout
        for &val in weights.iter() {
            weight_bytes.extend_from_slice(&val.to_le_bytes());
        }
        weight_specs.push(serde_json::json!({
            "name": format!("{}/kernel", layer_names[i]),
            "shape": [in_size, out_size],
            "dtype": "float32"
        }));

        // Biases
        for &val in biases.iter() {
            weight_bytes.extend_from_slice(&val.to_le_bytes());
        }
        weight_specs.push(serde_json::json!({
            "name": format!("{}/bias", layer_names[i]),
            "shape": [out_size],
            "dtype": "float32"
        }));
    }

    let model_topology = serde_json::json!({
        "class_name": "Sequential",
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "class_name": "Dense",
                    "config": {
                        "units": 256, "activation": "relu", "use_bias": true,
                        "name": "dense", "batch_input_shape": [null, 28],
                        "dtype": "float32"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "units": 64, "activation": "relu", "use_bias": true,
                        "name": "dense_1", "dtype": "float32"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "units": 4, "activation": "linear", "use_bias": true,
                        "name": "dense_2", "dtype": "float32"
                    }
                }
            ]
        }
    });

    let export = serde_json::json!({
        "modelTopology": model_topology,
        "weightSpecs": weight_specs,
        "weightData": weight_bytes,
        "meta": { "epsilon": agent.epsilon }
    });

    std::fs::write(filename, serde_json::to_string(&export).unwrap()).unwrap();
}

fn main() {
    let config = parse_args();
    let num_episodes = config.episodes;
    let print_every = config.print_every;
    let save_every = config.save_every;

    let mut agent = DQNAgent::new();
    let mut engine = SnakeEngine::new(GRID_SIZE);

    let mut max_score: i32 = 0;
    let mut best_avg: f32 = 0.0;
    let mut recent_scores: VecDeque<i32> = VecDeque::new();
    let start = Instant::now();

    println!("=== Snake DQN Trainer (Rust) ===");
    println!(
        "Grid: {}x{} | MLP 28→256→64→4 | Episodes: {} | DoubleDQN soft_tau=0.001 LR_decay",
        GRID_SIZE, GRID_SIZE, num_episodes
    );
    println!(
        "{:<10} {:<8} {:<8} {:<10} {:<10} {:<10} {:<8} {:<10}",
        "Episode", "Score", "Max", "Avg(100)", "Epsilon", "LR", "Buffer", "Time"
    );
    println!("{}", "-".repeat(78));

    for episode in 1..=num_episodes {
        engine.reset();
        let mut state = extract_features(&engine);
        loop {
            let action = agent.act(&state);
            let (reward, done) = engine.step(action);
            let next_state = extract_features(&engine);

            agent.remember(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            agent.step_and_train();
            state = next_state;

            if done {
                break;
            }
        }

        agent.end_episode();

        let score = engine.score;
        if score > max_score {
            max_score = score;
        }

        recent_scores.push_back(score);
        if recent_scores.len() > 100 {
            recent_scores.pop_front();
        }
        let avg = recent_scores.iter().sum::<i32>() as f32 / recent_scores.len() as f32;

        if recent_scores.len() >= 100 && avg > best_avg {
            best_avg = avg;
            export_model(&agent, "model_best.json");
        }

        if episode % print_every == 0 || episode == 1 {
            let elapsed = start.elapsed().as_secs();
            let mins = elapsed / 60;
            let secs = elapsed % 60;
            println!(
                "{:<10} {:<8} {:<8} {:<10.1} {:<10.4} {:<10.6} {:<8} {:02}:{:02}",
                episode,
                score,
                max_score,
                avg,
                agent.epsilon,
                agent.learning_rate,
                agent.buffer_len(),
                mins,
                secs
            );
        }

        if episode % save_every == 0 {
            let filename = format!("model_ep{}.json", episode);
            export_model(&agent, &filename);
            println!(">>> Saved: {} | Best avg: {:.1}", filename, best_avg);
        }
    }

    export_model(&agent, "model_final.json");
    println!(">>> Saved: model_final.json | Best avg: {:.1}", best_avg);
    println!("Done. Total time: {:?}", start.elapsed());
}
