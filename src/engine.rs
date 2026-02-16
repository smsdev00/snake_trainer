use rand::Rng;
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Right => (1, 0),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub const ACTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
];

pub struct SnakeEngine {
    pub grid_size: i32,
    pub snake: Vec<Point>,
    pub direction: Direction,
    pub food: Point,
    pub score: i32,
    pub game_over: bool,
    pub steps_without_food: i32,
}

impl SnakeEngine {
    pub fn new(grid_size: i32) -> Self {
        let mut engine = SnakeEngine {
            grid_size,
            snake: Vec::new(),
            direction: Direction::Right,
            food: Point { x: 0, y: 0 },
            score: 0,
            game_over: false,
            steps_without_food: 0,
        };
        engine.reset();
        engine
    }

    pub fn reset(&mut self) {
        let mid = self.grid_size / 2;
        self.snake = vec![
            Point { x: mid, y: mid },
            Point { x: mid - 1, y: mid },
            Point { x: mid - 2, y: mid },
        ];
        self.direction = Direction::Right;
        self.score = 0;
        self.game_over = false;
        self.steps_without_food = 0;
        self.food = self.spawn_food();
    }

    pub fn step(&mut self, action: usize) -> (f32, bool) {
        let dir = ACTIONS[action];
        if dir.opposite() != self.direction {
            self.direction = dir;
        }

        let head = self.snake[0];
        let prev_dist = (head.x - self.food.x).abs() + (head.y - self.food.y).abs();
        let prev_score = self.score;

        self.update();

        let reward;
        if self.game_over {
            reward = -10.0;
        } else if self.score > prev_score {
            reward = 10.0;
            self.steps_without_food = 0;
        } else {
            self.steps_without_food += 1;
            if self.steps_without_food > self.grid_size * self.grid_size {
                self.game_over = true;
                reward = -10.0;
            } else {
                let new_head = self.snake[0];
                let new_dist =
                    (new_head.x - self.food.x).abs() + (new_head.y - self.food.y).abs();
                let approach = if new_dist < prev_dist { 1.0 } else { -1.0 };

                // Preventive reward shaping (only kicks in when snake is big enough to matter)
                let snake_len = self.snake.len() as f32;
                let area = (self.grid_size * self.grid_size) as f32;

                let safety_bonus = if snake_len > area * 0.15 {
                    let reachable = self.flood_fill_from_head() as f32;
                    let _free = area - snake_len;

                    // Penalize if reachable space < snake length (trapped, can't fit)
                    let space_penalty = if reachable < snake_len {
                        -2.0
                    } else if reachable < snake_len * 1.5 {
                        -0.5
                    } else {
                        0.0
                    };

                    // Bonus for maintaining access to tail
                    let tail_bonus = if self.can_reach_tail() { 0.5 } else { -1.0 };

                    space_penalty + tail_bonus
                } else {
                    0.0
                };

                reward = approach + safety_bonus;
            }
        }

        (reward, self.game_over)
    }

    fn update(&mut self) {
        if self.game_over {
            return;
        }

        let (dx, dy) = self.direction.delta();
        let head = self.snake[0];
        let new_head = Point {
            x: head.x + dx,
            y: head.y + dy,
        };

        if new_head.x < 0
            || new_head.x >= self.grid_size
            || new_head.y < 0
            || new_head.y >= self.grid_size
        {
            self.game_over = true;
            return;
        }

        if self
            .snake
            .iter()
            .any(|s| s.x == new_head.x && s.y == new_head.y)
        {
            self.game_over = true;
            return;
        }

        self.snake.insert(0, new_head);

        if new_head.x == self.food.x && new_head.y == self.food.y {
            self.score += 10;
            self.food = self.spawn_food();
        } else {
            self.snake.pop();
        }
    }

    fn flood_fill_from_head(&self) -> u32 {
        let gs = self.grid_size;
        let head = self.snake[0];
        let occupied: HashSet<(i32, i32)> = self.snake.iter().map(|s| (s.x, s.y)).collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert((head.x, head.y));
        queue.push_back((head.x, head.y));
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

    /// BFS from head to tail (tail cell is walkable since it moves away)
    fn can_reach_tail(&self) -> bool {
        let gs = self.grid_size;
        let head = self.snake[0];
        let tail = self.snake[self.snake.len() - 1];

        // Occupied = all body EXCEPT tail (tail will move, so it's reachable)
        let mut occupied: HashSet<(i32, i32)> = self.snake.iter().map(|s| (s.x, s.y)).collect();
        occupied.remove(&(tail.x, tail.y));

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert((head.x, head.y));
        queue.push_back((head.x, head.y));

        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in &[(0i32, -1i32), (1, 0), (0, 1), (-1, 0)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx == tail.x && ny == tail.y {
                    return true;
                }
                if nx >= 0 && nx < gs && ny >= 0 && ny < gs
                    && !occupied.contains(&(nx, ny))
                    && !visited.contains(&(nx, ny))
                {
                    visited.insert((nx, ny));
                    queue.push_back((nx, ny));
                }
            }
        }

        false
    }

    fn spawn_food(&self) -> Point {
        let mut rng = rand::thread_rng();
        let mut free = Vec::new();
        for x in 0..self.grid_size {
            for y in 0..self.grid_size {
                if !self.snake.iter().any(|s| s.x == x && s.y == y) {
                    free.push(Point { x, y });
                }
            }
        }
        if free.is_empty() {
            return Point { x: 0, y: 0 };
        }
        free[rng.gen_range(0..free.len())]
    }
}
