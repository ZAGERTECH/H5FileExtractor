import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
                             QLabel, QTreeWidget, QTreeWidgetItem, QHeaderView,
                             QTreeWidgetItemIterator, QProgressDialog)
from PyQt5.QtCore import Qt


class H5DataMatrixExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" 帧序列数据提取器 (按列提取)")
        self.resize(900, 700)

        self.h5_file = None
        self.frames_group = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 顶部工具栏 ---
        top_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开 H5 文件")
        self.btn_open.clicked.connect(self.open_file)
        self.label_file_path = QLabel("当前未打开文件")
        self.label_file_path.setStyleSheet("color: gray;")

        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.label_file_path)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # --- 主体：数据字段选择树 ---
        self.label_info = QLabel("请在下方勾选你需要导出的数据字段 (基于第一帧自动识别)：")
        main_layout.addWidget(self.label_info)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["数据节点 (Sensor / Field)", "示例数值 (预览)", "数据类型"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        main_layout.addWidget(self.tree)

        # --- 底部：导出按钮 ---
        bottom_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("导出选中的列 (CSV)")
        self.btn_export_excel = QPushButton("导出选中的列 (Excel)")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_excel.setEnabled(False)

        self.btn_export_csv.clicked.connect(lambda: self.export_data(is_excel=False))
        self.btn_export_excel.clicked.connect(lambda: self.export_data(is_excel=True))

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_export_csv)
        bottom_layout.addWidget(self.btn_export_excel)
        main_layout.addLayout(bottom_layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 H5 文件", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
                                                   options=options)

        if file_path:
            try:
                if self.h5_file:
                    self.h5_file.close()

                self.h5_file = h5py.File(file_path, 'r')

                #  严格校验: 必须包含 frames 文件夹
                if 'frames' not in self.h5_file or not isinstance(self.h5_file['frames'], h5py.Group):
                    QMessageBox.warning(self, "结构校验失败", "该 H5 文件不包含 'frames' 根组，或者结构不符合预期！")
                    self.h5_file.close()
                    self.h5_file = None
                    return

                self.frames_group = self.h5_file['frames']
                frame_keys = sorted(self.frames_group.keys())

                if not frame_keys:
                    QMessageBox.warning(self, "无数据", "'frames' 文件夹为空！")
                    return

                self.label_file_path.setText(f"当前文件: {file_path} (共 {len(frame_keys)} 帧)")
                self.label_file_path.setStyleSheet("color: green;")

                #  读取第一帧，构建勾选树
                first_frame_id = frame_keys[0]
                self.build_schema_tree(first_frame_id)

                self.btn_export_csv.setEnabled(True)
                self.btn_export_excel.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取失败:\n{str(e)}")

    def build_schema_tree(self, frame_id):
        self.tree.clear()
        first_frame_node = self.frames_group[frame_id]

        # 递归遍历第一帧内的所有节点
        self._add_tree_nodes(self.tree, first_frame_node, "")

        # 强制全部折叠
        self.tree.collapseAll()

    def _add_tree_nodes(self, parent_ui_node, h5_node, current_path):
        for name, node in h5_node.items():
            # 记录它在 h5 中的相对路径，例如: can_info/Vehicle_Speed
            rel_path = f"{current_path}/{name}" if current_path else name

            if isinstance(node, h5py.Group):
                # 这是一个文件夹 (如 can_info)
                ui_node = QTreeWidgetItem(parent_ui_node, [name, "[组/目录]", ""])
                # 允许其拥有复选框，并且带有自动级联选中（点击父节点，子节点全选）
                ui_node.setFlags(ui_node.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                ui_node.setCheckState(0, Qt.Unchecked)
                # 递归里面的内容
                self._add_tree_nodes(ui_node, node, rel_path)

            elif isinstance(node, h5py.Dataset):
                # 这是一个具体的数据字段 (如 Vehicle_Speed)
                # 过滤掉高维数组(如图像), 只留标量或1D短数组用于CSV列导出
                # if len(node.shape) >= 2 or (len(node.shape) == 1 and node.shape[0] > 10):
                #     continue

                try:
                    # 抓取一个预览值
                    val = node[0] if len(node.shape) > 0 else node[()]
                    val_str = val.decode('utf-8', 'ignore') if isinstance(val, bytes) else str(val)
                except:
                    val_str = "无法预览"

                ui_node = QTreeWidgetItem(parent_ui_node, [name, val_str, str(node.dtype)])
                ui_node.setFlags(ui_node.flags() | Qt.ItemIsUserCheckable)
                ui_node.setCheckState(0, Qt.Unchecked)
                # 把 h5 的真实路径存入隐藏数据中，供导出时使用
                ui_node.setData(0, Qt.UserRole, rel_path)

    def export_data(self, is_excel=False):
        # 扫描树，收集用户勾选了哪些具体的底层字段
        selected_h5_paths = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.Checked and item.data(0, Qt.UserRole):
                selected_h5_paths.append(item.data(0, Qt.UserRole))
            iterator += 1

        if not selected_h5_paths:
            QMessageBox.warning(self, "未选择", "请至少勾选一个数据字段！")
            return

        # 让用户选择保存路径
        ext = "xlsx" if is_excel else "csv"
        file_filter = "Excel Files (*.xlsx)" if is_excel else "CSV Files (*.csv)"
        save_path, _ = QFileDialog.getSaveFileName(self, f"保存为主表 ({ext.upper()})", "", file_filter)

        if not save_path:
            return

        # 获取保存目录和基础文件名，用于分离图像和矩阵文件
        save_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        image_save_dir = os.path.join(save_dir, f"{base_name}_images")

        # 准备容器
        frame_keys = sorted(self.frames_group.keys())
        total_frames = len(frame_keys)

        all_rows_1d = []  # 用于存常规标量数据
        matrix_data_dict = {}  # 用于存矩阵数据 {col_name: [(frame_id, matrix_array), ...]}
        has_images = False  # 标记是否导出了图片

        progress = QProgressDialog("正在提取分类数据...", "取消", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)

        try:
            for i, fid in enumerate(frame_keys):
                if progress.wasCanceled():
                    break

                frame_node = self.frames_group[fid]
                row_dict_1d = {'Frame_ID': fid}  # 1D 主表第一列固定为帧号

                # 去当前帧里捞取用户勾选的数据
                for path in selected_h5_paths:
                    col_name = path.replace('/', '_')

                    try:
                        ds = frame_node[path]
                        shape_len = len(ds.shape)

                        # ======= case A：处理图像数据 (保存为 PNG) =======
                        if 'image' in col_name.lower() or shape_len >= 3:
                            img_array = ds[:]
                            if img_array.size > 0:
                                if not has_images:
                                    os.makedirs(image_save_dir, exist_ok=True)
                                    has_images = True

                                # HDF5存的图若为RGB，cv2保存需要转为BGR。这里假设底层存的是标准的图像矩阵
                                img_filename = os.path.join(image_save_dir, f"{fid}_{col_name}.png")
                                # 如果图像色彩不对，可以改成 cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(img_filename, img_array)
                            continue

                        # ======= case B：处理 2D 矩阵 (如 frame_info_L) =======
                        elif shape_len == 2:
                            mat_array = ds[:]
                            if col_name not in matrix_data_dict:
                                matrix_data_dict[col_name] = []
                            matrix_data_dict[col_name].append((fid, mat_array))
                            continue

                        # ======= case C：处理常规 1D 标量 / 字符串 =======
                        else:
                            val = ds[0] if shape_len > 0 else ds[()]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', 'ignore')
                            # 如果是一个小数组，强转为字符串存入单元格
                            elif isinstance(val, np.ndarray):
                                val = str(val.tolist())

                            row_dict_1d[col_name] = val

                    except Exception:
                        # 万一某一帧缺了这个数据
                        row_dict_1d[col_name] = None

                        # 如果这一帧除了 Frame_ID 还有其他 1D 数据，才加入主表
                if len(row_dict_1d) > 1:
                    all_rows_1d.append(row_dict_1d)

                progress.setValue(i + 1)

            # 开始写出各类文件
            progress.setLabelText("正在写入文件，请稍候...")
            export_summary = []

            # --- 写出常规 1D 主表 ---
            if all_rows_1d:
                df_1d = pd.DataFrame(all_rows_1d)
                if is_excel:
                    df_1d.to_excel(save_path, index=False)
                else:
                    df_1d.to_csv(save_path, index=False)
                export_summary.append("1D 主表导出成功")

            # --- 写出矩阵副表 (每个矩阵字段独立一个文件，帧之间空一行) ---
            for col_name, mat_list in matrix_data_dict.items():
                mat_save_path = os.path.join(save_dir, f"{base_name}_{col_name}.{ext}")
                giant_list = []

                for fid, mat in mat_list:
                    # 写入标识当前帧的表头 (占据第一列)
                    # 为了防止列数报错，用 None 填充补齐后面的列
                    padding_cols = [None] * (mat.shape[1] - 1 if mat.shape[1] > 1 else 0)
                    giant_list.append([f"Frame_ID: {fid}"] + padding_cols)

                    if mat.size > 0:
                        for row in mat:
                            giant_list.append(row.tolist())
                    else:
                        giant_list.append(["[Empty Matrix]"] + padding_cols)

                    # 每个矩阵之间空一行
                    giant_list.append([None] * (mat.shape[1] if mat.shape[1] > 0 else 1))

                # 将巨型列表转换为 DataFrame 一次性保存，兼容 CSV/Excel
                df_mat = pd.DataFrame(giant_list)
                if is_excel:
                    df_mat.to_excel(mat_save_path, index=False, header=False)
                else:
                    df_mat.to_csv(mat_save_path, index=False, header=False)
                export_summary.append(f"矩阵表 [{col_name}] 导出成功")

            if has_images:
                export_summary.append(f"图像序列已保存至文件夹: {base_name}_images")

            progress.setValue(total_frames)
            summary_text = "\n".join(export_summary)
            QMessageBox.information(self, "导出完成", f"跨帧数据提取完毕！\n\n【导出明细】:\n{summary_text}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出过程中发生错误:\n{str(e)}")

    def closeEvent(self, event):
        if self.h5_file:
            self.h5_file.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = H5DataMatrixExtractor()
    window.show()
    sys.exit(app.exec_())