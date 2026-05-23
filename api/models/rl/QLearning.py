import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback

class GridWorld:
    def __init__(self, width=8, height=8, start=(0,0), goal=(7,7), walls=None):
        self.width = width
        self.height = height
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.walls = set([tuple(w) for w in walls]) if walls else set()
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        x, y = self.state

        if action == 0: y -= 1
        elif action == 1: x += 1
        elif action == 2: y += 1
        elif action == 3: x -= 1

        # Check boundaries
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return self.state, -5, False # Wall crash penalty

        next_state = (x, y)

        # Check walls
        if next_state in self.walls:
            return self.state, -5, False # Wall crash penalty

        self.state = next_state

        # Check goal
        if self.state == self.goal:
            return self.state, 100, True # Goal reward

        return self.state, -1, False # Step penalty

def render_grid_frame(env, path, episode, is_final=False):
    """Renders the maze and the agent's path"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # Draw Grid
    for x in range(env.width + 1):
        ax.axvline(x - 0.5, color='gray', linestyle='-', alpha=0.3)
    for y in range(env.height + 1):
        ax.axhline(y - 0.5, color='gray', linestyle='-', alpha=0.3)

    # Draw Walls
    for wx, wy in env.walls:
        rect = plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, facecolor='#4b5563')
        ax.add_patch(rect)

    # Draw Start and Goal
    rect_start = plt.Rectangle((env.start[0] - 0.5, env.start[1] - 0.5), 1, 1, facecolor='#3b82f6', alpha=0.5)
    rect_goal = plt.Rectangle((env.goal[0] - 0.5, env.goal[1] - 0.5), 1, 1, facecolor='#22c55e', alpha=0.5)
    ax.add_patch(rect_start)
    ax.add_patch(rect_goal)
    ax.text(env.start[0], env.start[1], 'S', ha='center', va='center', fontweight='bold', color='black')
    ax.text(env.goal[0], env.goal[1], 'G', ha='center', va='center', fontweight='bold', color='black')

    # Draw Path
    if len(path) > 0:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color='#ef4444', linewidth=2, alpha=0.7, marker='o', markersize=4)
        # Agent current pos
        ax.scatter(path[-1][0], path[-1][1], s=150, c='#ef4444', edgecolors='white', zorder=5)

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5) # Invert Y so 0,0 is top left
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Agent Path - Episode {episode}", fontsize=14)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_q_table(env, q_table):
    """Renders the final policy (arrows) and Q-value heatmap"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # Create heatmap data (Max Q value for each state)
    heatmap = np.zeros((env.height, env.width))
    for x in range(env.width):
        for y in range(env.height):
            if (x, y) in env.walls:
                heatmap[y, x] = np.nan
            elif (x, y) == env.goal:
                heatmap[y, x] = 100
            else:
                heatmap[y, x] = np.max(q_table[x, y])

    cmap = plt.cm.viridis
    cmap.set_bad(color='#4b5563') # Walls are dark gray

    im = ax.imshow(heatmap, cmap=cmap, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Max Q-Value")

    # Draw Policy Arrows
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    arrow_dict = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}

    for x in range(env.width):
        for y in range(env.height):
            if (x, y) in env.walls or (x, y) == env.goal:
                continue

            best_action = np.argmax(q_table[x, y])
            dx, dy = arrow_dict[best_action]

            # Only draw arrow if the state has been visited/learned
            if np.max(q_table[x, y]) != 0:
                ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='white', ec='white', width=0.02)

    ax.text(env.start[0], env.start[1], 'S', ha='center', va='center', fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(env.goal[0], env.goal[1], 'G', ha='center', va='center', fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimal Policy (Q-Table)", fontsize=14)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_qlearning(data):
    try:
        grid_config = data.get('grid', {})
        width = int(grid_config.get('width', 8))
        height = int(grid_config.get('height', 8))
        start = grid_config.get('start', {'x': 0, 'y': 0})
        goal = grid_config.get('goal', {'x': 7, 'y': 7})
        walls = grid_config.get('walls',[])

        start_coord = (int(start['x']), int(start['y']))
        goal_coord = (int(goal['x']), int(goal['y']))
        walls_coords = [(int(w['x']), int(w['y'])) for w in walls]

        # Hyperparameters
        params = data.get('parameters', {})
        episodes = int(params.get('episodes', 500))
        alpha = float(params.get('alpha', 0.1))
        gamma = float(params.get('gamma', 0.9))
        epsilon = float(params.get('epsilon', 0.1))

        env = GridWorld(width, height, start_coord, goal_coord, walls_coords)

        # Initialize Q-Table: dimensions (width, height, 4 actions)
        q_table = np.zeros((width, height, 4))

        history = []
        reward_history =[]

        # Select episodes to animate (roughly 10 frames + final)
        frames_to_render = np.unique(np.linspace(0, episodes - 1, 10, dtype=int)).tolist()
        if episodes - 1 not in frames_to_render:
            frames_to_render.append(episodes - 1)

        success_count = 0

        # Training Loop
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            path = [state]
            done = False
            steps = 0
            max_steps = width * height * 2 # Prevent infinite loops

            while not done and steps < max_steps:
                x, y = state

                # Epsilon-Greedy Action Selection
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(q_table[x, y])

                next_state, reward, done = env.step(action)

                # Q-Learning Update Formula
                nx, ny = next_state
                old_value = q_table[x, y, action]
                next_max = np.max(q_table[nx, ny])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[x, y, action] = new_value

                state = next_state
                total_reward += reward
                steps += 1
                path.append(state)

            if done and state == env.goal:
                success_count += 1

            # Log for Recharts
            reward_history.append({
                "episode": ep + 1,
                "reward": float(total_reward),
                "steps": steps
            })

            # Render frame if needed
            if ep in frames_to_render:
                frame_img = render_grid_frame(env, path, ep + 1)
                history.append({
                    "episode": ep + 1,
                    "reward": float(total_reward),
                    "steps": steps,
                    "image": frame_img
                })

        final_q_table_img = render_q_table(env, q_table)

        return {
            'episodes': episodes,
            'success_rate': float(success_count / episodes),
            'final_reward': reward_history[-1]['reward'],
            'history': history,
            'reward_history': reward_history,
            'q_table_image': final_q_table_img
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_rl_sample_map(data):
    """Generates predefined maze layouts"""
    try:
        map_type = data.get('map_type', 'simple')
        width, height = 8, 8
        start = {'x': 0, 'y': 0}
        goal = {'x': 7, 'y': 7}
        walls =[]

        if map_type == 'corridor':
            # Create a winding path
            for i in range(1, 7):
                walls.append({'x': i, 'y': 2})
                walls.append({'x': 6 - i, 'y': 5})
        elif map_type == 'maze':
            # Create a complex maze
            maze_walls =[(1,1),(1,2),(1,3),(1,4),(1,5),(3,2),(3,3),(3,4),(3,5),(3,6),(5,1),(5,2),(5,3),(5,4),(5,5)]
            walls = [{'x': w[0], 'y': w[1]} for w in maze_walls]
        else: # simple
            # Just a simple wall in the middle
            walls =[{'x': 3, 'y': i} for i in range(2, 6)]
            walls.extend([{'x': 4, 'y': i} for i in range(2, 6)])

        return {
            'width': width,
            'height': height,
            'start': start,
            'goal': goal,
            'walls': walls
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}
