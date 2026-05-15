import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy.signal import find_peaks

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SpatialArrayAnalyzer:
    def __init__(self):
        self.c = 343  # Speed of sound (m/s)
        self.frequencies = np.logspace(2.3, 3.8, 50)  # 200Hz to 6kHz
        self.angles = np.linspace(0, 2*np.pi, 360)

    def create_array_geometry(self, array_type='FOA', radius=0.05):
        """Create microphone array geometry"""
        if array_type == 'FOA':
            # 4-channel First Order Ambisonics (tetrahedral)
            angles = np.array([0, 2*np.pi/3, 4*np.pi/3, np.pi])
            elevations = np.array([0, 0, 0, np.pi/2])
            positions = np.array([
                [radius * np.cos(angles[i]) * np.cos(elevations[i]),
                 radius * np.sin(angles[i]) * np.cos(elevations[i]),
                 radius * np.sin(elevations[i])] for i in range(4)
            ])
        else:  # HOA - 16 channel
            # Spherical array with 16 microphones
            n_mics = 16
            positions = []
            # Two rings of 6 mics each + 4 at poles
            for ring in [0.3, -0.3]:  # Two horizontal rings
                for i in range(6):
                    angle = i * 2 * np.pi / 6
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    z = ring * radius
                    positions.append([x, y, z])
            # Add 4 mics at different elevations
            for i in range(4):
                angle = i * np.pi / 2
                x = radius * 0.7 * np.cos(angle)
                y = radius * 0.7 * np.sin(angle)
                z = radius * 0.8 if i < 2 else -radius * 0.8
                positions.append([x, y, z])
            positions = np.array(positions)
        return positions

    def compute_steering_vector(self, positions, frequency, doa_angle):
        """Compute steering vector for given DOA"""
        direction = np.array([np.cos(doa_angle), np.sin(doa_angle), 0])
        k = 2 * np.pi * frequency / self.c
        steering_vector = np.exp(-1j * k * np.dot(positions, direction))
        return steering_vector

    def compute_beampattern(self, positions, frequency, target_angle):
        """Compute beampattern using MVDR beamforming"""
        # Create steering vector for target direction
        target_steering = self.compute_steering_vector(positions, frequency, target_angle)

        # Compute beampattern for all angles
        beampattern = np.zeros(len(self.angles))
        for i, angle in enumerate(self.angles):
            steering_vec = self.compute_steering_vector(positions, frequency, angle)
            # Normalized beampattern
            response = np.abs(np.vdot(target_steering, steering_vec))**2
            beampattern[i] = response / np.max(np.abs(target_steering)**2)

        return beampattern

def create_comprehensive_visualization():
    analyzer = SpatialArrayAnalyzer()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Array geometries
    foa_positions = analyzer.create_array_geometry('FOA')
    hoa_positions = analyzer.create_array_geometry('HOA')

    # 2. Frequency Response Analysis
    target_angle = np.pi/6  # 30 degrees
    foa_responses = []
    hoa_responses = []

    for freq in analyzer.frequencies:
        foa_bp = analyzer.compute_beampattern(foa_positions, freq, target_angle)
        hoa_bp = analyzer.compute_beampattern(hoa_positions, freq, target_angle)

        # Find main lobe width (3dB beamwidth)
        foa_peak_idx = np.argmax(foa_bp)
        hoa_peak_idx = np.argmax(hoa_bp)

        # Calculate 3dB beamwidth
        foa_3db = np.where(foa_bp >= 0.5 * np.max(foa_bp))[0]
        hoa_3db = np.where(hoa_bp >= 0.5 * np.max(hoa_bp))[0]

        foa_beamwidth = len(foa_3db) * 360 / len(analyzer.angles)
        hoa_beamwidth = len(hoa_3db) * 360 / len(analyzer.angles)

        foa_responses.append(foa_beamwidth)
        hoa_responses.append(hoa_beamwidth)

    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.semilogx(analyzer.frequencies, foa_responses, 'r-', linewidth=2, label='FOA (4 mics)', marker='o')
    ax3.semilogx(analyzer.frequencies, hoa_responses, 'b-', linewidth=2, label='HOA (16 mics)', marker='s')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('3dB Beamwidth (degrees)')
    ax3.set_title('Spatial Resolution vs Frequency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 180)

    plt.tight_layout()
    return fig

# Create the visualization
fig = create_comprehensive_visualization()
plt.show()