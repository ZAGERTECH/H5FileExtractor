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

    def _handle_image_data(self, ds, fid, col_name, image_save_dir, has_images):
        """Case A：处理图像数据，拦截假图，并根据 L/R 分文件夹保存为 PNG"""
        img_array = ds[:]

        # 拦截 1x1 或更小的假图片/占位图
        if img_array.size == 0 or (img_array.shape[0] <= 1 and img_array.shape[1] <= 1):
            return has_images

        # 判断属于左相机还是右相机
        sub_folder = "Other"  # 默认保底文件夹
        if col_name.endswith("_L") or "_L_" in col_name:
            sub_folder = "L"
        elif col_name.endswith("_R") or "_R_" in col_name:
            sub_folder = "R"

        # 拼接出最终的目标目录，例如: xxx_images/L
        target_dir = os.path.join(image_save_dir, sub_folder)

        # 创建目录并保存
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        img_filename = os.path.join(target_dir, f"{fid}_{col_name}.png")
        # 假设底层是RGB存的，OpenCV默认写BGR，如果颜色不对这里可以加个 cv2.cvtColor
        cv2.imwrite(img_filename, img_array)

        return True  # 只要保存了真实图片，就返回 True 标记

        return has_images

    def _handle_matrix_data(self, ds, fid, col_name, matrix_data_dict):
        """Case B：处理 2D 矩阵数据，记录到字典中等待批量导出"""
        mat_array = ds[:]
        if col_name not in matrix_data_dict:
            matrix_data_dict[col_name] = []
        matrix_data_dict[col_name].append((fid, mat_array))

    def _handle_scalar_data(self, ds, col_name, row_dict_1d, shape_len):
        """Case C：处理常规 1D 标量或字符串，直接填入当前行字典"""
        val = ds[0] if shape_len > 0 else ds[()]
        if isinstance(val, bytes):
            val = val.decode('utf-8', 'ignore')
        elif isinstance(val, np.ndarray):
            val = str(val.tolist())

        row_dict_1d[col_name] = val

    def export_data(self, is_excel=False):
        selected_h5_paths = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.Checked and item.data(0, Qt.UserRole):
                selected_h5_paths.append(item.data(0, Qt.UserRole))
            iterator += 1

        if not selected_h5_paths:
            QMessageBox.warning(self, "未选择", "请至少勾选一个数据字段。")
            return

        ext = "xlsx" if is_excel else "csv"
        file_filter = "Excel Files (*.xlsx)" if is_excel else "CSV Files (*.csv)"
        # 用户选择保存的基础路径和文件名
        user_choice_path, _ = QFileDialog.getSaveFileName(self, "选择保存位置及名称", "", file_filter)

        if not user_choice_path:
            return

        # 路径解析
        base_dir = os.path.dirname(user_choice_path)
        chosen_prefix = os.path.splitext(os.path.basename(user_choice_path))[0]
        h5_filename_no_ext = os.path.splitext(os.path.basename(self.h5_file.filename))[0]

        # 建立一级根目录: [H5文件名]_h5
        root_export_dir = os.path.join(base_dir, f"{h5_filename_no_ext}_h5")

        # 建立二级子目录
        common_data_dir = os.path.join(root_export_dir, f"{chosen_prefix}_common_data")
        frame_info_dir = os.path.join(root_export_dir, f"{chosen_prefix}_frame_info")
        image_save_dir = os.path.join(root_export_dir, f"{chosen_prefix}_images")

        # 确保目录存在
        os.makedirs(common_data_dir, exist_ok=True)

        # 准备容器
        frame_keys = sorted(self.frames_group.keys())
        total_frames = len(frame_keys)
        all_rows_1d = []
        matrix_data_dict = {}
        has_images = False

        progress = QProgressDialog("正在提取数据...", "取消", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)

        try:
            for i, fid in enumerate(frame_keys):
                if progress.wasCanceled():
                    break

                frame_node = self.frames_group[fid]
                row_dict_1d = {'Frame_ID': fid}

                for path in selected_h5_paths:
                    col_name = path.replace('/', '_')
                    try:
                        ds = frame_node[path]
                        shape_len = len(ds.shape)

                        if 'image' in col_name.lower() or shape_len >= 3:
                            # 传递新的图像根目录
                            has_images = self._handle_image_data(ds, fid, col_name, image_save_dir, has_images)
                        elif shape_len == 2:
                            self._handle_matrix_data(ds, fid, path, matrix_data_dict)
                        else:
                            self._handle_scalar_data(ds, col_name, row_dict_1d, shape_len)
                    except Exception:
                        row_dict_1d[col_name] = None

                if len(row_dict_1d) > 1:
                    all_rows_1d.append(row_dict_1d)
                progress.setValue(i + 1)

            export_summary = []

            # 写入 1D 主表到 common_data 文件夹
            if all_rows_1d:
                main_csv_path = os.path.join(common_data_dir, f"{chosen_prefix}.{ext}")
                df_1d = pd.DataFrame(all_rows_1d)
                if is_excel:
                    df_1d.to_excel(main_csv_path, index=False)
                else:
                    df_1d.to_csv(main_csv_path, index=False)
                export_summary.append(f"主表: {chosen_prefix}.{ext}")

            # 写入矩阵数据到 frame_info 文件夹
            if matrix_data_dict:
                os.makedirs(frame_info_dir, exist_ok=True)
                for h5_path, mat_list in matrix_data_dict.items():
                    dataset_name = h5_path.split('/')[-1]
                    mat_save_path = os.path.join(frame_info_dir, f"{dataset_name}.{ext}")

                    giant_list = []
                    max_cols = 0
                    for fid, mat in mat_list:
                        if mat.size > 0:
                            max_cols = max(max_cols, mat.shape[1] if len(mat.shape) > 1 else mat.shape[0])

                    header = ["Frame_ID"] + [str(idx) for idx in range(max_cols)]
                    for fid, mat in mat_list:
                        if mat.size > 0:
                            if len(mat.shape) == 1:
                                mat = mat.reshape(1, -1)
                            for row in mat:
                                row_list = row.tolist()
                                padded_row = row_list + [None] * (max_cols - len(row_list))
                                giant_list.append([fid] + padded_row)

                    if giant_list:
                        df_mat = pd.DataFrame(giant_list, columns=header)
                        if is_excel:
                            df_mat.to_excel(mat_save_path, index=False)
                        else:
                            df_mat.to_csv(mat_save_path, index=False)
                        export_summary.append(f"矩阵: {dataset_name}.{ext}")

            if has_images:
                export_summary.append("图像序列已分类导出")

            progress.setValue(total_frames)
            summary_text = "\n".join(export_summary)
            QMessageBox.information(self, "导出完成",
                                    f"已在以下位置生成数据结构：\n{root_export_dir}\n\n详情：\n{summary_text}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")


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