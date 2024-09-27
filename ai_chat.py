from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon,QActionGroup
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLineEdit, QToolBar, QStatusBar, QMenuBar, QFileDialog,
    QColorDialog, QFontDialog, QMessageBox
)

class AIChatApp(QMainWindow):
    """Main Window for AI Chat Application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat Application")
        self.setGeometry(100, 100, 800, 600)
        
        self._theme = "dark"
        self._corner_shape = "rounded"
        
        # Central widget for the main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        
        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText("Type your message here...")
        
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.handle_send_message)
        
        # Layout setup
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.chat_history)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)
        self.layout.addLayout(input_layout)
        
        # Setup toolbar, menubar, and status bar
        self._setup_toolbar()
        self._setup_menubar()
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        
    def _setup_toolbar(self):
        """Setup the top toolbar."""
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        clear_chat_action = QAction(QIcon("icons:clear_24dp.svg"), "Clear Chat", self)
        clear_chat_action.triggered.connect(self.clear_chat_history)
        
        change_theme_action = QAction(QIcon("icons:contrast_24dp.svg"), "Change Theme", self)
        change_theme_action.triggered.connect(self.change_theme)
        
        toolbar.addAction(clear_chat_action)
        toolbar.addAction(change_theme_action)
        
    def _setup_menubar(self):
        """Setup the menubar with options."""
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)
        
        # Dialog actions
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Open...", self.open_file_dialog)
        file_menu.addAction("Save As...", self.save_file_dialog)
        
        dialog_menu = menubar.addMenu("&Dialogs")
        dialog_menu.addAction("Open Color Dialog", self.open_color_dialog)
        dialog_menu.addAction("Open Font Dialog", self.open_font_dialog)
        
    def handle_send_message(self):
        """Handle sending and receiving chat messages."""
        user_message = self.user_input.text()
        if user_message.strip():
            self.chat_history.append(f"<b>You:</b> {user_message}")
            self.user_input.clear()
            response = self.generate_ai_response(user_message)
            self.chat_history.append(f"<b>AI:</b> {response}")
    
    def generate_ai_response(self, message):
        """Generate a placeholder AI response (can integrate with actual model)."""
        # Placeholder logic; integrate your AI model here
        return f"I received your message: {message}"
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history.clear()
    
    def change_theme(self):
        """Switch between light and dark themes."""
        if self._theme == "dark":
            self._theme = "light"
        else:
            self._theme = "dark"
        # Implement theme change logic using qdarktheme or other theme managers.
    
    def open_file_dialog(self):
        """Open a file dialog to load chat history."""
        QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)")
    
    def save_file_dialog(self):
        """Open a save dialog to save chat history."""
        QFileDialog.getSaveFileName(self, "Save File As", "", "Text Files (*.txt)")
    
    def open_color_dialog(self):
        """Open a color dialog."""
        QColorDialog.getColor(self)
    
    def open_font_dialog(self):
        """Open a font dialog."""
        QFontDialog.getFont(self)
    
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    chat_app = AIChatApp()
    chat_app.show()
    sys.exit(app.exec())