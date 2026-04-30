import os
import sys
import multiprocessing
import h5py
import ctypes
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
                             QLabel, QTreeWidget, QTreeWidgetItem, QHeaderView,
                             QTreeWidgetItemIterator, QProgressDialog, QInputDialog,
                             QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QIcon
import openpyxl
from ExportWorker import ExportWorker


class H5DataMatrixExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H5FileExtractor")
        self.resize(900, 700)

        self.h5_file_paths = []
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        self.btn_open = QPushButton("加载 H5 文件")
        self.btn_open.clicked.connect(self.open_files)
        self.label_file_path = QLabel("当前未加载文件")
        self.label_file_path.setStyleSheet("color: gray;")

        self.btn_about = QPushButton("关于")
        self.btn_about.clicked.connect(self.show_about_dialog)

        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.label_file_path)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_about)
        main_layout.addLayout(top_layout)

        self.label_info = QLabel("请在下方勾选需要导出的数据字段 (基于首个文件自动识别)：")
        main_layout.addWidget(self.label_info)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["数据节点 (Sensor / Field)", "示例数值 (预览)", "数据类型"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        main_layout.addWidget(self.tree)

        thread_layout = QHBoxLayout()
        self.label_thread = QLabel("解析线程数:")
        self.spin_thread = QSpinBox()
        self.spin_thread.setMinimum(1)
        max_cores = os.cpu_count() or 1
        self.spin_thread.setMaximum(max_cores)
        self.spin_thread.setValue(max_cores)

        thread_layout.addWidget(self.label_thread)
        thread_layout.addWidget(self.spin_thread)
        thread_layout.addStretch()
        main_layout.addLayout(thread_layout)

        bottom_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("批量导出选中的元素 (CSV)")
        self.btn_export_excel = QPushButton("批量导出选中的元素 (Excel)")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_excel.setEnabled(False)

        self.btn_export_csv.clicked.connect(lambda: self.export_batch_data(is_excel=False))
        self.btn_export_excel.clicked.connect(lambda: self.export_batch_data(is_excel=True))

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_export_csv)
        bottom_layout.addWidget(self.btn_export_excel)
        main_layout.addLayout(bottom_layout)

    def show_about_dialog(self):
        about_text = (
            "<h2 align='center'>H5FileExtractor</h2>"
            "<table>"
            "<tr>"
            "<td><b>版本：</b></td>"
            "<td>Ver.260430</td>"
            "</tr>"
            "<tr>"
            "<td><b>作者：</b></td>"
            "<td>sbhinx</td>"
            "</tr>"
            "<tr>"
            "<td><b>描述：</b></td>"
            "<td>用于批量解析 HDF5 格式的路测数据</td>"
            "</tr>"
            "<tr>"
            "<td><b>Github：</b></td>"
            "<td><a href='https://github.com/ZAGERTECH/H5FileExtractor'>https://github.com/ZAGERTECH/H5FileExtractor</a></td>"
            "</tr>"
        )
        QMessageBox.about(self, "关于", about_text)

    def open_files(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择 H5 文件(支持多个)", "",
                                                     "HDF5 Files (*.h5 *.hdf5);;All Files (*)", options=options)

        if file_paths:
            self.h5_file_paths = file_paths
            first_file = file_paths[0]

            try:
                with h5py.File(first_file, 'r') as f:
                    if 'frames' not in f or not isinstance(f['frames'], h5py.Group):
                        QMessageBox.warning(self, "结构校验失败", "首个 H5 文件不包含 'frames' 根组！")
                        self.h5_file_paths = []
                        return

                    frames_group = f['frames']
                    frame_keys = sorted(frames_group.keys())

                    if not frame_keys:
                        QMessageBox.warning(self, "无数据", "首个文件的 'frames' 文件夹为空！")
                        return

                    self.label_file_path.setText(
                        f"已加载 {len(file_paths)} 个文件 (树结构参照: {os.path.basename(first_file)})")
                    self.label_file_path.setStyleSheet("color: green;")

                    first_frame_id = frame_keys[0]
                    self.build_schema_tree(frames_group[first_frame_id])

                self.btn_export_csv.setEnabled(True)
                self.btn_export_excel.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"首个文件读取失败:\n{str(e)}")

    def build_schema_tree(self, first_frame_node):
        self.tree.clear()

        # 新增母树节点，开启用户复选和自动级联（Tristate）状态
        root_node = QTreeWidgetItem(self.tree, ["全选 / 取消全选", "", ""])
        root_node.setFlags(root_node.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        root_node.setCheckState(0, Qt.Unchecked)

        # 将第一帧的节点内容全部挂载到母节点下，而不是直接挂在 self.tree 下
        self._add_tree_nodes(root_node, first_frame_node, "")

        # 默认展开母节点，折叠内部子节点保持清爽
        root_node.setExpanded(True)

    def _add_tree_nodes(self, parent_ui_node, h5_node, current_path):
        for name, node in h5_node.items():
            rel_path = f"{current_path}/{name}" if current_path else name

            if isinstance(node, h5py.Group):
                ui_node = QTreeWidgetItem(parent_ui_node, [name, "[组/目录]", ""])
                ui_node.setFlags(ui_node.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                ui_node.setCheckState(0, Qt.Unchecked)
                ui_node.setExpanded(False)
                self._add_tree_nodes(ui_node, node, rel_path)

            elif isinstance(node, h5py.Dataset):
                try:
                    val = node[0] if len(node.shape) > 0 else node[()]
                    val_str = val.decode('utf-8', 'ignore') if isinstance(val, bytes) else str(val)
                except:
                    val_str = "无法预览"

                ui_node = QTreeWidgetItem(parent_ui_node, [name, val_str, str(node.dtype)])
                ui_node.setFlags(ui_node.flags() | Qt.ItemIsUserCheckable)
                ui_node.setCheckState(0, Qt.Unchecked)
                ui_node.setData(0, Qt.UserRole, rel_path)

    def export_batch_data(self, is_excel=False):
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
        output_base_dir = QFileDialog.getExistingDirectory(self, "选择批量导出的目标文件夹")
        if not output_base_dir:
            return

        chosen_prefix, ok = QInputDialog.getText(self, "命名前缀", "请输入导出文件的前缀名 (如 drv_data):",
                                                 text="drv_data")
        if not ok or not chosen_prefix.strip():
            return
        chosen_prefix = chosen_prefix.strip()

        total_frames = 0
        valid_files = []
        for fp in self.h5_file_paths:
            try:
                with h5py.File(fp, 'r') as f:
                    if 'frames' in f:
                        keys = sorted(f['frames'].keys())
                        total_frames += len(keys)
                        valid_files.append((fp, keys))
            except:
                pass

        if total_frames == 0:
            return

        self.btn_export_csv.setEnabled(False)
        self.btn_export_excel.setEnabled(False)

        self.progress = QProgressDialog("正在后台提取数据...", "取消", 0, total_frames, self)
        self.progress.setWindowTitle("导出任务")
        self.progress.resize(300, 150)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setValue(0)

        # 禁用自动重置和自动关闭，确保它能撑到弹窗出现
        self.progress.setAutoReset(False)
        self.progress.setAutoClose(False)

        max_workers = self.spin_thread.value()

        self.worker = ExportWorker(
            valid_files=valid_files,
            selected_h5_paths=selected_h5_paths,
            output_base_dir=output_base_dir,
            chosen_prefix=chosen_prefix,
            ext=ext,
            is_excel=is_excel,
            max_workers=max_workers
        )

        self.worker.progress_updated.connect(self.progress.setValue)
        # 绑定状态信号到进度条的标签文字上
        # 收到禁用信号后，直接把取消按钮设为 None (即隐藏)
        self.worker.toggle_cancel_btn.connect(
            lambda state: [btn.setEnabled(state) for btn in self.progress.findChildren(QPushButton)]
        )
        self.worker.status_updated.connect(self.progress.setLabelText)
        self.worker.finished_successfully.connect(self.on_export_success)
        self.worker.error_occurred.connect(self.on_export_error)
        self.worker.export_cancelled.connect(self.on_export_cancel)
        self.progress.canceled.connect(self.worker.cancel)

        self.worker.start()

    def on_export_success(self, count, output_dir):
        self.btn_export_csv.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self.progress.close()
        QMessageBox.information(self, "批量导出完成", f"已成功处理 {count} 个文件。\n已输出至:\n{output_dir}")

    def on_export_error(self, error_msg):
        self.btn_export_csv.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self.progress.close()
        QMessageBox.critical(self, "错误", f"批量导出崩溃:\n{error_msg}")

    def on_export_cancel(self):
        self.btn_export_csv.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self.progress.close()
        QMessageBox.warning(self, "已取消", "后台数据提取任务已被中止。")

def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容开发环境和 PyInstaller 打包环境 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 告诉 Windows 这是一个独立的应用程序，切断与 python.exe 的图标默认绑定
    try:
        myappid = 'zagertech.h5extractor.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    active_highlight_color = palette.color(QPalette.Active, QPalette.Highlight)
    palette.setColor(QPalette.Inactive, QPalette.Highlight, active_highlight_color)
    app.setPalette(palette)

    # 设置全局应用程序图标（这会自动应用到主窗口以及所有弹出的子窗口/对话框）
    app.setWindowIcon(QIcon(resource_path("resource/logo.png")))

    window = H5DataMatrixExtractor()
    window.show()
    sys.exit(app.exec_())