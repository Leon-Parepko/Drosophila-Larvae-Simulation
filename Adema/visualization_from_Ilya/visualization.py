import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pandas as pd

import tempfile
import os
import shutil


class data:
    def __init__(self, data_path, w_path: str, chunk_size=500):
        self.path = data_path
        self.w_path = w_path
        self.chunk_size = chunk_size
        self.chunks_dataframe = pd.read_csv(self.path, chunksize=chunk_size)
        self.w = pd.read_csv(self.w_path).to_numpy(np.float32)

    def update_chunk(
        self,
        chunk: pd.DataFrame,
        ctx: "context",
        last_positions,
        dt=0.01,
        save_positions_every=1,
    ):
        activity = chunk.to_numpy(np.float32)
        positions = ctx.simulate_clasters(
            activity, self.w, last_positions, dt, save_positions_every
        )
        return positions

    def get_clasters(
        self,
        output_chunks_directory: str,
        start_ind=0,
        end_ind=-1,
        last_positions=None,
        dt=0.01,
        save_positions_every=1,
        pushing_force=None,
        pulling_force=None,
    ):
        """
        Обрабатывает чанки данных активности, обновляет позиции для каждого чанка и сохраняет результаты в CSV-файлы.
        Аргументы:
            output_chunks_directory (str): Директория, куда будут сохранены выходные CSV-файлы для каждого чанка.
            start_ind (int, необязательный): Индекс первого обрабатываемого чанка. По умолчанию 0.
            end_ind (int, необязательный): Индекс после последнего обрабатываемого чанка. Если -1, обрабатываются все до конца. По умолчанию -1.
            last_positions (np.ndarray, необязательный): Начальные позиции для обновления. Если None, генерируются случайные позиции. По умолчанию None.
            dt (float, необязательный): Шаг времени для обновления. По умолчанию 0.01.
            save_positions_every (int, необязательный): Частота (в тиках), с которой сохраняются позиции. По умолчанию 1.
        Возвращает:
            List[str]: Список путей к сохранённым CSV-файлам для каждого обработанного чанка.
        """
        activity_dim = self.w.shape[0]
        paths_tochunks = []
        c = context(
            activity_dim=activity_dim,
            pulling_force=pulling_force,
            pushing_force=pushing_force,
        )
        if last_positions is None:
            last_positions = np.random.uniform(-2, 2, (activity_dim * 2)).astype(
                np.float32
            )
        for n, chunk in enumerate(self.chunks_dataframe):
            if start_ind <= n and (n < end_ind or end_ind == -1):
                positions = self.update_chunk(
                    chunk,
                    c,
                    last_positions,
                    dt=dt,
                    save_positions_every=save_positions_every,
                )
                last_positions = positions[-1]
                print("chunk", n, "generated")

                name = f"chunk{n} shape_{activity_dim}x2 every_{save_positions_every}_tick.csv"
                pd.DataFrame(
                    positions, columns=[i for i in range(activity_dim * 2)]
                ).to_csv(pp := os.path.join(output_chunks_directory, name))
                paths_tochunks.append(pp)
                print("chunk", n, "saved to", pp)
            elif n >= end_ind:
                break
        return paths_tochunks

    def connect_chunks(self, paths_to_chunks, output_path, chunksize=10000):
        header_written = False
        current_index = 0  # Глобальный индекс для итогового файла
        for chunk_path in paths_to_chunks:
            for chunk in pd.read_csv(chunk_path, chunksize=chunksize, index_col=0):
                chunk.reset_index(drop=True, inplace=True)  # Сбросить локальные индексы
                # Присваиваем новые уникальные индексы для итогового файла
                chunk.index = range(current_index, current_index + len(chunk))
                chunk.to_csv(
                    output_path, mode="a", header=not header_written, index=False
                )
                header_written = True
                current_index += len(chunk)

    def save_clusters(self, name, positions):
        pd.DataFrame(positions).to_csv(name)

    def process_and_merge_chunks(
        self, output_merged_path, chunks_dir=None, **get_clasters_kwargs
    ):
        """
        Генерирует чанки в указанную директорию (или временную), затем объединяет их в один файл.
        :param output_merged_path: Путь для итогового объединённого файла.
        :param chunks_dir: Путь для хранения чанков (если None — создаётся временная директория).
        :param get_clasters_kwargs: Дополнительные параметры для get_clasters.
        """
        with open(output_merged_path, "w") as f:
            f.write("")
            print(output_merged_path, "cleared")
        temp_dir = None
        if chunks_dir is None:
            temp_dir = tempfile.mkdtemp()
            chunks_dir = temp_dir
        try:
            paths = self.get_clasters(chunks_dir, **get_clasters_kwargs)
            self.connect_chunks(paths, output_merged_path)
            print("Merged file saved to", output_merged_path)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)


class context:
    def __init__(
        self, activity_dim, cl_context=None, pulling_force=None, pushing_force=None
    ):
        self.pulling_force = pulling_force
        self.pushing_force = pushing_force
        self.activity_dim = activity_dim
        if cl_context is None:
            cl_context = cl.create_some_context(0)
        self.ctx = cl_context
        self.queqe = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, self.source_code).build()

    def ccb(self, a: np.ndarray, dtype=np.float32):
        return cl.Buffer(
            self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=a.astype(dtype)
        )

    def cdb(self, a: np.ndarray, dtype=np.float32):
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=a.astype(dtype))

    @property
    def source_code(self):
        with open("./kernels.cl", "r") as j:
            code = j.read()
        if self.pulling_force is not None:
            code = (
                f""" 
#define PULLING_FORCE {self.pulling_force}\n"""
                + code
            )
        if self.pushing_force is not None:
            code = (
                f"""
#define PUSHING_FORCE {self.pushing_force}\n"""
                + code
            )
        return code

    def simulate_clasters(self, activity, w, positions, dt, save_positions_every=1):
        output_positions = []
        flattened_positions = (
            np.reshape(positions, positions.shape[0] * positions.shape[1])
            if len(positions.shape) > 1
            else positions
        )

        ab = self.cdb(np.reshape(activity, activity.shape[0] * activity.shape[1]))
        w = self.ccb(np.reshape(w, w.shape[0] * w.shape[1]))
        positions_ = self.cdb(flattened_positions)
        next_positions_ = self.cdb(np.empty_like(flattened_positions))
        dt_ = self.ccb(np.array(dt))
        activity_size_ = self.ccb(np.array(activity.shape), np.int32)

        for ti in range(len(activity)):
            t = self.ccb(np.array(ti), np.int32)
            self.program.simulate_clasters(
                self.queqe,
                (self.activity_dim,),
                None,
                ab,
                w,
                positions_,
                next_positions_,
                activity_size_,
                t,
                dt_,
            )
            positions_, next_positions_ = next_positions_, positions_
            if ti % save_positions_every == 0:
                op = np.empty_like(flattened_positions)
                cl.enqueue_copy(self.queqe, op, positions_)
                output_positions.append(op)

        return output_positions
