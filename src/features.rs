use crate::engine::{Direction, SnakeEngine};
use std::collections::{HashSet, VecDeque};

fn relative_dirs(dir: Direction) -> (Direction, Direction, Direction) {
    match dir {
        Direction::Up => (Direction::Up, Direction::Right, Direction::Left),
        Direction::Right => (Direction::Right, Direction::Down, Direction::Up),
        Direction::Down => (Direction::Down, Direction::Left, Direction::Right),
        Direction::Left => (Direction::Left, Direction::Up, Direction::Down),
    }
}

fn is_collision(x: i32, y: i32, engine: &SnakeEngine) -> bool {
    if x < 0 || x >= engine.grid_size || y < 0 || y >= engine.grid_size {
        return true;
    }
    engine.snake.iter().any(|s| s.x == x && s.y == y)
}

/// Ray-cast: distance to first obstacle (wall or body) in given direction, normalized by grid size
fn ray_distance(engine: &SnakeEngine, dx: i32, dy: i32) -> f32 {
    let head = engine.snake[0];
    let gs = engine.grid_size as f32;
    let mut x = head.x + dx;
    let mut y = head.y + dy;
    let mut dist = 1;

    while x >= 0 && x < engine.grid_size && y >= 0 && y < engine.grid_size
        && !engine.snake.iter().any(|s| s.x == x && s.y == y)
    {
        x += dx;
        y += dy;
        dist += 1;
    }

    dist as f32 / gs
}

/// Flood fill from a starting point, reusing a pre-built occupied set
fn flood_fill_from(
    start_x: i32,
    start_y: i32,
    gs: i32,
    occupied: &HashSet<(i32, i32)>,
) -> u32 {
    if start_x < 0 || start_x >= gs || start_y < 0 || start_y >= gs
        || occupied.contains(&(start_x, start_y))
    {
        return 0;
    }

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
            if nx >= 0 && nx < gs && ny >= 0 && ny < gs
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

/// 28 features: original 22 + ray-cast (3) + directional flood fill (3)
pub fn extract_features(engine: &SnakeEngine) -> Vec<f32> {
    let head = engine.snake[0];
    let tail = engine.snake[engine.snake.len() - 1];
    let dir = engine.direction;
    let gs = engine.grid_size;
    let gsf = gs as f32;

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

    // Ray-cast distance in each relative direction (normalized)
    let ray_straight = ray_distance(engine, sdx, sdy);
    let ray_right = ray_distance(engine, rdx, rdy);
    let ray_left = ray_distance(engine, ldx, ldy);

    // Direction one-hot
    let dir_up = if dir == Direction::Up { 1.0 } else { 0.0 };
    let dir_right_f = if dir == Direction::Right { 1.0 } else { 0.0 };
    let dir_down = if dir == Direction::Down { 1.0 } else { 0.0 };
    let dir_left_f = if dir == Direction::Left { 1.0 } else { 0.0 };

    // Food relative
    let food_up = if engine.food.y < head.y { 1.0 } else { 0.0 };
    let food_right = if engine.food.x > head.x { 1.0 } else { 0.0 };
    let food_down = if engine.food.y > head.y { 1.0 } else { 0.0 };
    let food_left = if engine.food.x < head.x { 1.0 } else { 0.0 };

    // Wall distances (normalized)
    let wall_up = head.y as f32 / gsf;
    let wall_down = (gs - 1 - head.y) as f32 / gsf;
    let wall_left = head.x as f32 / gsf;
    let wall_right = (gs - 1 - head.x) as f32 / gsf;

    // Snake length (normalized)
    let snake_length = engine.snake.len() as f32 / (gsf * gsf);

    // Build occupied set once for all flood fills
    let occupied: HashSet<(i32, i32)> = engine.snake.iter().map(|s| (s.x, s.y)).collect();

    // Global flood fill ratio
    let total_free = (gs * gs) as f32 - engine.snake.len() as f32;
    let reachable = flood_fill_from(head.x, head.y, gs, &occupied) as f32;
    let flood_ratio = if total_free > 0.0 { reachable / total_free } else { 0.0 };

    // Directional flood fill: reachable space from cell in each relative direction
    let flood_straight = flood_fill_from(head.x + sdx, head.y + sdy, gs, &occupied) as f32;
    let flood_right_f = flood_fill_from(head.x + rdx, head.y + rdy, gs, &occupied) as f32;
    let flood_left_f = flood_fill_from(head.x + ldx, head.y + ldy, gs, &occupied) as f32;
    let flood_max = total_free.max(1.0);
    let flood_straight_n = flood_straight / flood_max;
    let flood_right_n = flood_right_f / flood_max;
    let flood_left_n = flood_left_f / flood_max;

    // Tail relative direction (sign)
    let tail_dx = (tail.x - head.x).signum() as f32;
    let tail_dy = (tail.y - head.y).signum() as f32;

    vec![
        danger_straight, danger_right, danger_left,
        danger_straight2, danger_right2, danger_left2,
        ray_straight, ray_right, ray_left,
        dir_up, dir_right_f, dir_down, dir_left_f,
        food_up, food_right, food_down, food_left,
        wall_up, wall_right, wall_down, wall_left,
        snake_length,
        flood_ratio,
        flood_straight_n, flood_right_n, flood_left_n,
        tail_dx, tail_dy,
    ]
}
