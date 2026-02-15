use crate::engine::{Direction, SnakeEngine};
use std::collections::{HashSet, VecDeque};

const RELATIVE: [(Direction, Direction, Direction); 4] = [
    // (straight, right, left) for each direction
    (Direction::Up, Direction::Right, Direction::Left),       // UP
    (Direction::Right, Direction::Down, Direction::Up),       // RIGHT
    (Direction::Down, Direction::Left, Direction::Right),     // DOWN
    (Direction::Left, Direction::Up, Direction::Down),        // LEFT
];

fn relative_dirs(dir: Direction) -> (Direction, Direction, Direction) {
    match dir {
        Direction::Up => RELATIVE[0],
        Direction::Right => RELATIVE[1],
        Direction::Down => RELATIVE[2],
        Direction::Left => RELATIVE[3],
    }
}

fn is_collision(x: i32, y: i32, engine: &SnakeEngine) -> bool {
    if x < 0 || x >= engine.grid_size || y < 0 || y >= engine.grid_size {
        return true;
    }
    engine.snake.iter().any(|s| s.x == x && s.y == y)
}

fn flood_fill(start_x: i32, start_y: i32, engine: &SnakeEngine) -> u32 {
    let gs = engine.grid_size;
    if start_x < 0 || start_x >= gs || start_y < 0 || start_y >= gs {
        return 0;
    }

    let occupied: HashSet<(i32, i32)> = engine.snake.iter().map(|s| (s.x, s.y)).collect();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert((start_x, start_y));
    queue.push_back((start_x, start_y));
    let mut count = 0u32;

    while let Some((x, y)) = queue.pop_front() {
        count += 1;
        for &(dx, dy) in &[(0i32, -1i32), (1, 0), (0, 1), (-1, 0)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0
                && nx < gs
                && ny >= 0
                && ny < gs
                && !occupied.contains(&(nx, ny))
                && !visited.contains(&(nx, ny))
            {
                visited.insert((nx, ny));
                queue.push_back((nx, ny));
            }
        }
    }

    count
}

/// 22 features matching the JS version exactly
pub fn extract_features(engine: &SnakeEngine) -> Vec<f32> {
    let head = engine.snake[0];
    let tail = engine.snake[engine.snake.len() - 1];
    let dir = engine.direction;
    let gs = engine.grid_size as f32;

    let (straight, right, left) = relative_dirs(dir);
    let (sdx, sdy) = straight.delta();
    let (rdx, rdy) = right.delta();
    let (ldx, ldy) = left.delta();

    // Danger 1 step
    let danger_straight = if is_collision(head.x + sdx, head.y + sdy, engine) { 1.0 } else { 0.0 };
    let danger_right = if is_collision(head.x + rdx, head.y + rdy, engine) { 1.0 } else { 0.0 };
    let danger_left = if is_collision(head.x + ldx, head.y + ldy, engine) { 1.0 } else { 0.0 };

    // Danger 2 steps
    let danger_straight2 = if is_collision(head.x + sdx * 2, head.y + sdy * 2, engine) { 1.0 } else { 0.0 };
    let danger_right2 = if is_collision(head.x + rdx * 2, head.y + rdy * 2, engine) { 1.0 } else { 0.0 };
    let danger_left2 = if is_collision(head.x + ldx * 2, head.y + ldy * 2, engine) { 1.0 } else { 0.0 };

    // Direction one-hot
    let dir_up = if dir == Direction::Up { 1.0 } else { 0.0 };
    let dir_right = if dir == Direction::Right { 1.0 } else { 0.0 };
    let dir_down = if dir == Direction::Down { 1.0 } else { 0.0 };
    let dir_left = if dir == Direction::Left { 1.0 } else { 0.0 };

    // Food relative
    let food_up = if engine.food.y < head.y { 1.0 } else { 0.0 };
    let food_right = if engine.food.x > head.x { 1.0 } else { 0.0 };
    let food_down = if engine.food.y > head.y { 1.0 } else { 0.0 };
    let food_left = if engine.food.x < head.x { 1.0 } else { 0.0 };

    // Wall distances (normalized)
    let wall_up = head.y as f32 / gs;
    let wall_down = (engine.grid_size - 1 - head.y) as f32 / gs;
    let wall_left = head.x as f32 / gs;
    let wall_right = (engine.grid_size - 1 - head.x) as f32 / gs;

    // Snake length (normalized)
    let snake_length = engine.snake.len() as f32 / (gs * gs);

    // Flood fill ratio
    let total_free = (engine.grid_size * engine.grid_size) as f32 - engine.snake.len() as f32;
    let reachable = flood_fill(head.x, head.y, engine) as f32;
    let flood_ratio = if total_free > 0.0 { reachable / total_free } else { 0.0 };

    // Tail relative direction (sign)
    let tail_dx = (tail.x - head.x).signum() as f32;
    let tail_dy = (tail.y - head.y).signum() as f32;

    vec![
        danger_straight, danger_right, danger_left,
        danger_straight2, danger_right2, danger_left2,
        dir_up, dir_right, dir_down, dir_left,
        food_up, food_right, food_down, food_left,
        wall_up, wall_right, wall_down, wall_left,
        snake_length,
        flood_ratio,
        tail_dx, tail_dy,
    ]
}
