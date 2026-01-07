"""
Demo GIF Generator for Three-Body Simulator
Captures frames and creates an animated GIF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
sys.path.insert(0, '.')

from nbody_simulation_advanced import (
    SimulationConfig, SimulationState, NBodySimulator,
    generate_initial_conditions, rk4_step_adaptive
)

def generate_demo_gif(output_path='demo.gif', duration_seconds=10, fps=30):
    """Generate a demo GIF with zoomed view"""
    
    # ズームした設定
    config = SimulationConfig()
    config.display_range = 0.8  # ズームイン
    
    simulator = NBodySimulator(config)
    state = simulator.state
    
    # ウォームアップ: 最初の100ステップを録画前に実行
    print("Warming up simulation...")
    for _ in range(100):
        simulator.step(config.steps_per_frame)
        simulator.update_trails()
    print("Warmup complete!")
    
    # プロット設定
    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    
    display_range = config.display_range
    ax.set_xlim(-display_range, display_range)
    ax.set_ylim(-display_range, display_range)
    ax.set_zlim(-display_range, display_range)
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')
    ax.set_title('Three-Body Chaos Dance', color='white', fontsize=16, fontweight='bold')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:state.n_bodies]
    
    # プロットオブジェクト
    bodies = []
    trails = []
    for i in range(state.n_bodies):
        body, = ax.plot([], [], [], 'o', color=colors[i], markersize=12,
                       markeredgecolor='white', markeredgewidth=2)
        bodies.append(body)
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.6, linewidth=2)
        trails.append(trail)
    
    total_frames = duration_seconds * fps
    
    def update(frame):
        # シミュレーション進行（境界チェックなし - 連続録画用）
        simulator.step(config.steps_per_frame)
        simulator.update_trails()
        for i in range(state.n_bodies):
            x, y, z = state.positions[i]
            bodies[i].set_data([x], [y])
            bodies[i].set_3d_properties([z])
            
            if state.trail_history[i]:
                trail_arr = np.array(state.trail_history[i])
                trails[i].set_data(trail_arr[:, 0], trail_arr[:, 1])
                trails[i].set_3d_properties(trail_arr[:, 2])
        
        # 自動回転
        ax.view_init(elev=20, azim=30 + frame * 0.5)
        
        if frame % 30 == 0:
            print(f"Frame {frame}/{total_frames} ({100*frame//total_frames}%)")
        
        return bodies + trails
    
    print(f"Generating {total_frames} frames for {duration_seconds}s GIF...")
    
    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, interval=1000//fps)
    
    print(f"Saving to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    
    print(f"Done! Saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_demo_gif('demo.gif', duration_seconds=10, fps=20)
