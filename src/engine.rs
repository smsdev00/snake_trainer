use rand::Rng;

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
                let new_dist = (new_head.x - self.food.x).abs() + (new_head.y - self.food.y).abs();
                reward = if new_dist < prev_dist { 1.0 } else { -1.0 };
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
