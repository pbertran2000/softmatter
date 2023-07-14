import numpy as np
import matplotlib.pyplot as plt

# List of trajectory file names
file_names = ['T=300_3', 'T=290_3', 'T=298_3']

conversion_factor = 2 / 100
corr_factor_lists = []  # List to store correlation factor lists for each file

for file_name in file_names:
    data = []

    with open(file_name, 'r') as file:
        num_atoms = int(file.readline().strip())
        for line in file:
            columns = line.strip().split()[1:]  # Skip the first column
            if len(columns) == 3:
                data.append([float(value) for value in columns])

    num_frames = int(len(data)/num_atoms)  # Assuming each frame has num_atoms * 3 lines
    print(num_frames)

    mu_ini = []
    corr_factor_list = []
    temps = np.linspace(0, num_frames * conversion_factor, num_frames)

    for frame in range(num_frames):
        frame_vectors = []

        for i in range(frame * num_atoms, (frame + 1) * num_atoms, 3):
            molecules = data[i:i+3]  # Get the H1, H2, OH pack
            H2 = np.array(molecules[2])
            H1 = np.array(molecules[1])
            OH = np.array(molecules[0])

            vector1 = H2 - OH
            vector2 = H1 - OH

            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)

            bisector = norm_vector2 * vector1 + norm_vector1 * vector2
            unitary_bisector = bisector / np.linalg.norm(bisector)

            if frame == 0:
                mu_ini.append(unitary_bisector)

            frame_vectors.append(unitary_bisector)

        mu_tot = [frame_vectors[i] * mu_ini[i] for i in range(len(mu_ini))]
        mu_prom = np.mean(mu_tot)

        norm_factor = [mu_ini[i] * mu_ini[i] for i in range(len(mu_ini))]
        norm_factor_mean = np.mean(norm_factor)

        corr_factor = mu_prom / norm_factor_mean
        corr_factor_list.append(corr_factor)

    corr_factor_lists.append(corr_factor_list)
    plt.plot(temps, corr_factor_list)

# Plotting

plt.xlabel('t')
plt.ylabel('Correlation Factor')
plt.ylim(0,1)
plt.legend(file_names)
plt.show()