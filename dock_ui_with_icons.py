# filename: dock_ui_with_tabs_and_icon_menu.py
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
import sys
from PyQt6.QtWidgets import (
    QDockWidget, QMainWindow, QApplication,QTextEdit, QVBoxLayout, QWidget, QTabWidget, 
    QMenu, QPushButton, QToolBar, QStatusBar, QMenuBar
)

class DockUI:
    """The ui class of dock window."""

    def setup_ui(self, win: QWidget) -> None:
        """Set up ui with tabs and dock widgets."""
        # Create main window and tab widget
        self.main_win = QMainWindow()
        self.tab_widget = QTabWidget()
        
        # Create central widget
        self.central_edit = QTextEdit("This is the central widget.")
        self.main_win.setCentralWidget(self.central_edit)
        
        # Initialize dock widgets
        self.create_dock_widgets()

        # Create toolbar with icons to add new dockable widgets
        self.setup_toolbar()

        # Set up menu
        self.setup_menu()

        # Add dockable widget areas
        self.add_dockable_widgets()

        # Layout
        layout = QVBoxLayout(win)
        layout.addWidget(self.main_win)
        layout.setContentsMargins(0, 0, 0, 0)

    def create_dock_widgets(self):
        """Initialize the dock widgets for the UI."""
        self.left_dock = QDockWidget("Chat", self.main_win)
        self.right_dock = QDockWidget("Gallery", self.main_win)
        self.top_dock = QDockWidget("Image Gen", self.main_win)
        self.bottom_dock = QDockWidget("Vector Storage", self.main_win)

        # Assign a QTextEdit to each dock widget (for demonstration purposes)
        self.left_dock.setWidget(QTextEdit("Chat functionality here."))
        self.right_dock.setWidget(QTextEdit("Gallery widget here."))
        self.top_dock.setWidget(QTextEdit("Image generation here."))
        self.bottom_dock.setWidget(QTextEdit("Vector storage here."))

    def setup_toolbar(self):
        """Create toolbar with icons to add new dockable widgets."""
        self.toolbar = QToolBar("Dock Widgets", self.main_win)
        
        # Add action icons
        self.add_chat_action = QAction(QIcon("chat_icon.png"), "Add Chat Dock", self.main_win)
        self.add_image_gen_action = QAction(QIcon("image_gen_icon.png"), "Add Image Gen Dock", self.main_win)
        self.add_gallery_action = QAction(QIcon("gallery_icon.png"), "Add Gallery Dock", self.main_win)
        self.add_vector_storage_action = QAction(QIcon("vector_icon.png"), "Add Vector Storage Dock", self.main_win)

        # Connect actions
        self.add_chat_action.triggered.connect(self.add_chat_dock)
        self.add_image_gen_action.triggered.connect(self.add_image_gen_dock)
        self.add_gallery_action.triggered.connect(self.add_gallery_dock)
        self.add_vector_storage_action.triggered.connect(self.add_vector_storage_dock)

        # Add actions to the toolbar
        self.toolbar.addAction(self.add_chat_action)
        self.toolbar.addAction(self.add_image_gen_action)
        self.toolbar.addAction(self.add_gallery_action)
        self.toolbar.addAction(self.add_vector_storage_action)
        self.main_win.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

    def setup_menu(self):
        """Set up menu bar with options."""
        self.menu_bar = QMenuBar(self.main_win)
        self.main_win.setMenuBar(self.menu_bar)

        # Create file menu
        file_menu = self.menu_bar.addMenu("File")

        # Add some example actions
        exit_action = QAction("Exit", self.main_win)
        exit_action.triggered.connect(self.main_win.close)
        file_menu.addAction(exit_action)

    def add_dockable_widgets(self):
        """Add dock widgets to the main window."""
        self.main_win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)
        self.main_win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
        self.main_win.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.top_dock)
        self.main_win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)

    def add_chat_dock(self):
        """Add new Chat dock dynamically."""
        new_chat_dock = QDockWidget("New Chat Dock", self.main_win)
        new_chat_dock.setWidget(QTextEdit("New chat dock added."))
        self.main_win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, new_chat_dock)

    def add_image_gen_dock(self):
        """Add new Image Gen dock dynamically."""
        new_img_gen_dock = QDockWidget("New Image Gen Dock", self.main_win)
        new_img_gen_dock.setWidget(QTextEdit("New image gen dock added."))
        self.main_win.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, new_img_gen_dock)

    def add_gallery_dock(self):
        """Add new Gallery dock dynamically."""
        new_gallery_dock = QDockWidget("New Gallery Dock", self.main_win)
        new_gallery_dock.setWidget(QTextEdit("New gallery dock added."))
        self.main_win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, new_gallery_dock)

    def add_vector_storage_dock(self):
        """Add new Vector Storage dock dynamically."""
        new_vector_dock = QDockWidget("New Vector Storage Dock", self.main_win)
        new_vector_dock.setWidget(QTextEdit("New vector storage dock added."))
        self.main_win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, new_vector_dock)

class MainApp(QWidget):
    """Main application class to demonstrate usage of DockUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Dockable Tabs with Icon Menu")
        self.setGeometry(100, 100, 1200, 800)

        # Create an instance of DockUI and set up the UI
        self.dock_ui = DockUI()
        self.dock_ui.setup_ui(self)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())