"""
N-body simulation physics module in Mojo.

This module provides high-performance gravitational physics calculations.
"""

from math import sqrt
from collections import List


@fieldwise_init
struct Vec3(Copyable, Movable):
    """A 3D vector struct."""

    var x: Float64
    var y: Float64
    var z: Float64

    fn __copyinit__(out self, existing: Self):
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z

    fn __moveinit__(out self, deinit existing: Self):
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z

    fn __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    fn __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __mul__(self, scalar: Float64) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    fn length_squared(self) -> Float64:
        return self.x * self.x + self.y * self.y + self.z * self.z

    fn length(self) -> Float64:
        return sqrt(self.length_squared())


fn compute_accelerations(
    positions: List[Vec3],
    masses: List[Float64],
    softening: Float64,
    g: Float64,
) -> List[Vec3]:
    """
    Compute gravitational accelerations for all bodies.

    Uses Plummer softening: F = Gm1m2 / (r^2 + eps^2)^(3/2).
    """
    var n = len(positions)
    var eps2 = softening * softening
    var accelerations = List[Vec3](capacity=n)

    for _ in range(n):
        accelerations.append(Vec3(0.0, 0.0, 0.0))

    for i in range(n):
        var acc = Vec3(0.0, 0.0, 0.0)
        for j in range(n):
            if i != j:
                var r_ij = positions[j] - positions[i]
                var r2 = r_ij.length_squared() + eps2
                var r = sqrt(r2)
                var inv_r3 = 1.0 / (r2 * r)
                var factor = g * masses[j] * inv_r3
                acc = acc + r_ij * factor
        accelerations[i] = acc.copy()

    return accelerations^


fn compute_min_distance(positions: List[Vec3]) -> Float64:
    """Compute minimum distance between any two bodies."""
    var n = len(positions)
    var min_dist: Float64 = 1e30

    for i in range(n):
        for j in range(i + 1, n):
            var dist = (positions[j] - positions[i]).length()
            if dist < min_dist:
                min_dist = dist

    return min_dist


fn adaptive_timestep(
    positions: List[Vec3],
    base_dt: Float64,
    min_dt: Float64,
    max_dt: Float64,
) -> Float64:
    """Compute adaptive timestep based on minimum distance."""
    var min_dist = compute_min_distance(positions)
    var factor = min_dist / 0.3
    if factor > 1.0:
        factor = 1.0
    var dt = base_dt * factor
    if dt < min_dt:
        return min_dt
    if dt > max_dt:
        return max_dt
    return dt


fn rk4_step(
    var positions: List[Vec3],
    var velocities: List[Vec3],
    masses: List[Float64],
    softening: Float64,
    dt: Float64,
    g: Float64,
) -> Tuple[List[Vec3], List[Vec3]]:
    """
    Perform one RK4 integration step.

    Returns new positions and velocities.
    """
    var n = len(positions)

    # k1
    var k1_v = compute_accelerations(positions, masses, softening, g)

    # k2
    var pos_k2 = List[Vec3](capacity=n)
    var vel_k2 = List[Vec3](capacity=n)
    for i in range(n):
        pos_k2.append(positions[i] + velocities[i] * (dt * 0.5))
        vel_k2.append(velocities[i] + k1_v[i] * (dt * 0.5))
    var k2_v = compute_accelerations(pos_k2, masses, softening, g)

    # k3
    var pos_k3 = List[Vec3](capacity=n)
    var vel_k3 = List[Vec3](capacity=n)
    for i in range(n):
        pos_k3.append(positions[i] + vel_k2[i] * (dt * 0.5))
        vel_k3.append(velocities[i] + k2_v[i] * (dt * 0.5))
    var k3_v = compute_accelerations(pos_k3, masses, softening, g)

    # k4
    var pos_k4 = List[Vec3](capacity=n)
    var vel_k4 = List[Vec3](capacity=n)
    for i in range(n):
        pos_k4.append(positions[i] + vel_k3[i] * dt)
        vel_k4.append(velocities[i] + k3_v[i] * dt)
    var k4_v = compute_accelerations(pos_k4, masses, softening, g)

    # Final update
    var new_positions = List[Vec3](capacity=n)
    var new_velocities = List[Vec3](capacity=n)
    var dt6 = dt / 6.0

    for i in range(n):
        var dr = (velocities[i] + vel_k2[i] * 2.0 + vel_k3[i] * 2.0 + vel_k4[i]) * dt6
        var dv = (k1_v[i] + k2_v[i] * 2.0 + k3_v[i] * 2.0 + k4_v[i]) * dt6
        new_positions.append(positions[i] + dr)
        new_velocities.append(velocities[i] + dv)

    return (new_positions^, new_velocities^)


fn compute_energy(
    positions: List[Vec3],
    velocities: List[Vec3],
    masses: List[Float64],
    softening: Float64,
    g: Float64,
) -> Float64:
    """Compute total energy (kinetic + potential)."""
    var n = len(positions)
    var eps2 = softening * softening

    # Kinetic energy
    var ke: Float64 = 0.0
    for i in range(n):
        ke += 0.5 * masses[i] * velocities[i].length_squared()

    # Potential energy
    var pe: Float64 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            var r2 = (positions[j] - positions[i]).length_squared()
            pe -= g * masses[i] * masses[j] / sqrt(r2 + eps2)

    return ke + pe


from python import Python, PythonObject

fn read_line(sys: PythonObject) raises -> String:
    """Read a line from stdin."""
    var line = sys.stdin.readline()
    return String(String(line).strip())

fn run_ipc_server() raises:
    """Run in IPC server mode: read state, compute, write state."""
    print("READY") # Signal to Python that we are ready
    
    # We'll use Python interop for robust I/O for now
    var sys = Python.import_module("sys")
    
    while True:
        try:
            var line_obj = sys.stdin.readline()
            var line_str = String(line_obj)
            if not line_str:
                break
                
            var text = String(line_str.strip())
            if text == "EXIT":
                break
            
            # Protocol: 
            # Input: N DT SOFTENING G STEPS
            # Input: MASSES (space separated)
            # Input: POS_X POS_Y POS_Z (per body, one line)
            # Input: VEL_X VEL_Y VEL_Z (per body, one line)
            
            var params = text.split(" ")
            var n = atol(params[0])
            var dt = atof(params[1])
            var softening = atof(params[2])
            var g = atof(params[3])
            var steps = 1
            if len(params) > 4:
                steps = atol(params[4])
            
            var masses_line = String(String(sys.stdin.readline()).strip())
            var masses_str = masses_line.split(" ")
            var masses = List[Float64](capacity=n)
            for i in range(n):
                masses.append(atof(masses_str[i]))
                
            var positions = List[Vec3](capacity=n)
            var positions_line = String(String(sys.stdin.readline()).strip())
            var pos_parts = positions_line.split(" ")
            for i in range(n):
                positions.append(Vec3(atof(pos_parts[i*3]), atof(pos_parts[i*3+1]), atof(pos_parts[i*3+2])))
                
            var velocities = List[Vec3](capacity=n)
            var velocities_line = String(String(sys.stdin.readline()).strip())
            var vel_parts = velocities_line.split(" ")
            for i in range(n):
                velocities.append(Vec3(atof(vel_parts[i*3]), atof(vel_parts[i*3+1]), atof(vel_parts[i*3+2])))
            
            # Compute steps (Batch Processing)
            var current_pos = positions^
            var current_vel = velocities^
            
            for _ in range(steps):
                # We interpret rk4_step as taking owned arguments or we provide copies.
                # Since List is not implicitly copyable, we must be explicit.
                var result = rk4_step(current_pos.copy(), current_vel.copy(), masses, softening, dt, g)
                
                # Assign new values. Since 'result' owns the lists, calling copy() is safe.
                # Using copy() avoids complex move semantics with tuples for now.
                current_pos = result[0].copy()
                current_vel = result[1].copy()
                
            var new_pos = current_pos^
            var new_vel = current_vel^
            
            # Output:
            # POS_X POS_Y POS_Z ...
            # VEL_X VEL_Y VEL_Z ...
            
            var out_pos = String("")
            for i in range(n):
                if i > 0: out_pos += " "
                out_pos += String(new_pos[i].x) + " " + String(new_pos[i].y) + " " + String(new_pos[i].z)
            print(out_pos)
            
            var out_vel = String("")
            for i in range(n):
                if i > 0: out_vel += " "
                out_vel += String(new_vel[i].x) + " " + String(new_vel[i].y) + " " + String(new_vel[i].z)
            print(out_vel)
            
            # Flush stdout to ensure Python gets it immediately
            sys.stdout.flush()
            
        except e:
            # On error, try to print it and continue or exit
            print("ERROR: " + String(e))
            sys.stdout.flush()
            break

fn main() raises:
    """
    Main entry point.
    If 'ipc' argument is provided, run in IPC server mode.
    Otherwise run benchmark.
    """
    from sys import argv
    var args = argv()
    
    if len(args) > 1 and args[1] == "ipc":
        run_ipc_server()
        return

    print("N-Body Physics Benchmark (Mojo)")
    print("=" * 40)

    # Initialize 3-body system (Figure-8)
    var positions = List[Vec3]()
    positions.append(Vec3(0.97000436, -0.24308753, 0.0))
    positions.append(Vec3(-0.97000436, 0.24308753, 0.0))
    positions.append(Vec3(0.0, 0.0, 0.0))

    var velocities = List[Vec3]()
    velocities.append(Vec3(0.466203685, 0.43236573, 0.0))
    velocities.append(Vec3(0.466203685, 0.43236573, 0.0))
    velocities.append(Vec3(-0.93240737, -0.86473146, 0.0))

    var masses = List[Float64]()
    masses.append(1.0)
    masses.append(1.0)
    masses.append(1.0)

    var softening: Float64 = 0.001
    var g: Float64 = 1.0
    var dt: Float64 = 0.001

    # Benchmark: run 10000 steps
    var steps = 10000
    print("Running", steps, "simulation steps...")

    for _ in range(steps):
        var result = rk4_step(positions^, velocities^, masses, softening, dt, g)
        positions = result[0].copy()
        velocities = result[1].copy()

    var energy = compute_energy(positions, velocities, masses, softening, g)
    print("Final energy:", energy)
    print("Final position[0]:", positions[0].x, positions[0].y, positions[0].z)
    print("Benchmark complete!")
