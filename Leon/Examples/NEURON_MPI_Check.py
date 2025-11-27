from neuron import h
from NEURON_Sim_Wrapper import Network
import pickle

# --- MPI инициализация ---
pc = h.ParallelContext()
rank = int(pc.id())     # номер текущего процесса
nhost = int(pc.nhost()) # общее количество процессов
print(rank)
if rank == 0:
    print(f"🚀 Запущено {nhost} MPI процессов")

# --- Список нейронов ---
neuron_list = [
    '7055857', '1805418', '14260575', '5835799', '10160250',
    '7840203', '5019924', '13986477', '10167078', '7982896',
    '4119387', '17591442', '4227544', '10495502', '8069478',
    '3913629', '11279244', '16846805', '8980589', '3664102'
]

# --- Автоматическое разбиение нейронов между MPI-процессами ---
def split_list(lst, n):
    """Делит список lst на n частей (по возможности равномерно)."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

neuron_chunks = split_list(neuron_list, nhost)
my_neurons = neuron_chunks[rank]

print(f"[Rank {rank}] моделирует нейроны: {my_neurons}")

# --- Создание и настройка сети ---
net = Network(my_neurons)
net.load_graphs()
net.build_sections()
net.connect_morphology()
net.build_synapses()

# --- Настройка записи и стимулов ---
net.setup_recording(neurons=my_neurons)

# Подать стимул только на первый нейрон (только rank=0 делает стимуляцию)
if rank == 0:
    net.setup_stimulus(neurons=['7055857'], start_time=10, duration=50, amplitude=1.5)

# --- Запуск симуляции ---
print(f"[Rank {rank}] запускает симуляцию...")
t, voltages = net.run(duration=200)
print(f"[Rank {rank}] симуляция завершена.")

# --- Сбор данных со всех процессов ---
local_result = {"rank": rank, "neurons": my_neurons, "t": t, "voltages": voltages}
all_results = pc.py_allgather(local_result)

# --- Только rank 0 сохраняет результаты ---
if rank == 0:
    with open("results_mpi.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("✅ Все результаты сохранены в results_mpi.pkl")

pc.barrier()
pc.done()
h.quit()