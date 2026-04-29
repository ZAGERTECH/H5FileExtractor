import multiprocessing
import os
import shutil

from PyQt5.QtCore import  QThread, pyqtSignal
import concurrent.futures
import cv2
import h5py
import numpy as np
import pandas as pd

def process_frames_chunk(filepath, chunk_fids, selected_h5_paths, image_save_dir, cancel_event):
    """子进程处理任务，增加 cancel_event 监听以支持瞬间阻断"""
    chunk_rows_1d = []
    chunk_matrix_dict = {}
    has_images = False

    with h5py.File(filepath, 'r', swmr=True) as f:
        if 'frames' not in f:
            return chunk_rows_1d, chunk_matrix_dict, has_images

        frames_group = f['frames']
        for fid in chunk_fids:
            # 核心修改：在处理每一帧之前，检查主进程是否发出了取消信号
            if cancel_event.is_set():
                return chunk_rows_1d, chunk_matrix_dict, has_images

            if fid not in frames_group:
                continue
            frame_node = frames_group[fid]
            row_dict_1d = {'Frame_ID': fid}

            for path in selected_h5_paths:
                col_name = path.replace('/', '_')
                try:
                    ds = frame_node[path]
                    shape_len = len(ds.shape)

                    if 'image' in col_name.lower() or shape_len >= 3:
                        img_array = ds[:]
                        if img_array.size > 0 and (img_array.shape[0] > 1 or img_array.shape[1] > 1):
                            sub_folder = "Other"
                            if col_name.endswith("_L") or "_L_" in col_name:
                                sub_folder = "L"
                            elif col_name.endswith("_R") or "_R_" in col_name:
                                sub_folder = "R"

                            target_dir = os.path.join(image_save_dir, sub_folder)
                            os.makedirs(target_dir, exist_ok=True)
                            img_filename = os.path.join(target_dir, f"{fid}_{col_name}.png")
                            cv2.imwrite(img_filename, img_array)
                            has_images = True

                    elif shape_len == 2:
                        mat_array = ds[:]
                        if path not in chunk_matrix_dict:
                            chunk_matrix_dict[path] = []
                        chunk_matrix_dict[path].append((fid, mat_array))

                    else:
                        val = ds[0] if shape_len > 0 else ds[()]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', 'ignore')
                        elif isinstance(val, np.ndarray):
                            val = str(val.tolist())
                        row_dict_1d[col_name] = val

                except Exception:
                    row_dict_1d[col_name] = None

            if len(row_dict_1d) > 1:
                chunk_rows_1d.append(row_dict_1d)

    return chunk_rows_1d, chunk_matrix_dict, has_images

class ExportWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished_successfully = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)
    export_cancelled = pyqtSignal()

    def __init__(self, valid_files, selected_h5_paths, output_base_dir, chosen_prefix, ext, is_excel, max_workers):
        super().__init__()
        self.valid_files = valid_files
        self.selected_h5_paths = selected_h5_paths
        self.output_base_dir = output_base_dir
        self.chosen_prefix = chosen_prefix
        self.ext = ext
        self.is_excel = is_excel
        self.max_workers = max_workers
        self.is_cancelled = False

        # 核心修改：创建一个跨进程共享的事件标志
        self.manager = multiprocessing.Manager()
        self.cancel_event = self.manager.Event()

    def run(self):
        current_frame_count = 0
        success_files_count = 0

        try:
            for fp, frame_keys in self.valid_files:
                if self.is_cancelled:
                    break

                h5_filename_no_ext = os.path.splitext(os.path.basename(fp))[0]
                root_export_dir = os.path.join(self.output_base_dir, f"{h5_filename_no_ext}_h5")

                if os.path.exists(root_export_dir):
                    shutil.rmtree(root_export_dir)

                common_data_dir = os.path.join(root_export_dir, f"{self.chosen_prefix}_common_data")
                frame_info_dir = os.path.join(root_export_dir, f"{self.chosen_prefix}_frame_info")
                image_save_dir = os.path.join(root_export_dir, f"{self.chosen_prefix}_images")

                # 重新创建基础目录
                os.makedirs(common_data_dir, exist_ok=True)

                all_rows_1d = []
                matrix_data_dict = {}

                chunk_size = 500
                chunks = [frame_keys[i:i + chunk_size] for i in range(0, len(frame_keys), chunk_size)]

                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # 将共享的 cancel_event 传递给所有子进程
                    futures = {
                        executor.submit(process_frames_chunk, fp, chunk, self.selected_h5_paths, image_save_dir,
                                        self.cancel_event): chunk
                        for chunk in chunks
                    }

                    for future in concurrent.futures.as_completed(futures):
                        if self.is_cancelled:
                            executor.shutdown(wait=False)
                            self.export_cancelled.emit()
                            return

                        chunk_fids = futures[future]
                        try:
                            c_rows_1d, c_mat_dict, _ = future.result()
                            all_rows_1d.extend(c_rows_1d)

                            for path, mat_list in c_mat_dict.items():
                                if path not in matrix_data_dict:
                                    matrix_data_dict[path] = []
                                matrix_data_dict[path].extend(mat_list)

                        except Exception as e:
                            print(f"解析块出现错误: {e}")

                        current_frame_count += len(chunk_fids)
                        self.progress_updated.emit(current_frame_count)

                all_rows_1d.sort(key=lambda x: x['Frame_ID'])
                for path in matrix_data_dict:
                    matrix_data_dict[path].sort(key=lambda x: x[0])

                if all_rows_1d:
                    main_csv_path = os.path.join(common_data_dir, f"{self.chosen_prefix}.{self.ext}")
                    df_1d = pd.DataFrame(all_rows_1d)
                    if self.is_excel:
                        df_1d.to_excel(main_csv_path, index=False)
                    else:
                        df_1d.to_csv(main_csv_path, index=False)

                if matrix_data_dict:
                    os.makedirs(frame_info_dir, exist_ok=True)
                    for h5_path, mat_list in matrix_data_dict.items():
                        dataset_name = h5_path.split('/')[-1]
                        mat_save_path = os.path.join(frame_info_dir, f"{dataset_name}.{self.ext}")

                        giant_list = []
                        max_cols = 0
                        for fid, mat in mat_list:
                            if mat.size > 0:
                                max_cols = max(max_cols, mat.shape[1] if len(mat.shape) > 1 else mat.shape[0])

                        header = ["Frame_ID"] + [str(idx) for idx in range(max_cols)]
                        for fid, mat in mat_list:
                            if mat.size > 0:
                                if len(mat.shape) == 1: mat = mat.reshape(1, -1)
                                for row in mat:
                                    row_list = row.tolist()
                                    padded_row = row_list + [None] * (max_cols - len(row_list))
                                    giant_list.append([fid] + padded_row)

                        if giant_list:
                            df_mat = pd.DataFrame(giant_list, columns=header)
                            if self.is_excel:
                                df_mat.to_excel(mat_save_path, index=False)
                            else:
                                df_mat.to_csv(mat_save_path, index=False)

                success_files_count += 1

            if not self.is_cancelled:
                self.finished_successfully.emit(success_files_count, self.output_base_dir)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self.is_cancelled = True
        # 核心修改：拉响全局警报，所有子进程读取到此信号会立即退出循环
        self.cancel_event.set()