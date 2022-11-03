
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout, QDateEdit, QHBoxLayout, QSizePolicy, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtCore import QSize, Qt, QDateTime, QDate
from PyQt5.QtGui import QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from functools import partial

# Only needed for access to command line arguments
import sys
from utils import get_df

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = get_df()
        self.setWindowTitle("Hack-QC - IUVO-AI")
        self.selectedError = 0
        self.layout = QGridLayout()

        calendar_layout = self.displayCalendar()
        self.layout.addLayout(calendar_layout, 0, 1)
        self.displayTitle(0,0)
        
        self.displayData(1, 0)
        self.displayMenu(2,1)
 
        self.displayInformation(2, 0)
        self.displayErrors(1,1)
        
        # method called when signal emitted


        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 8)
        self.layout.setRowStretch(2, 2)

        self.setFixedSize(QSize(1920, 1080))
        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
    
    def displayTitle(self, row, col):
        title = QLabel()
        title.setText("E.A.U. Erreur Automatique Update fait par <font color=teal>IUVO-AI</font>")
        title.setFont(QFont('Arial', 30))
        title.setAutoFillBackground(True)
        title.setStyleSheet("background-color: white;"
                             "border-style: solid;"
                             "border-width: 5px;"
                             "border-color: gray;"
                             "border-radius: 3px")
        title.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(title, row, col)

    def displayData(self, row, col):
        plotlayout = QVBoxLayout()
        temp_var = self.dateedit.date() 
        date = temp_var.toPyDate()
        self.daily = self.data.query(f'Year == {date.year} and Month == {date.month} and Day == {date.day}')
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        # Date Brutte_aval
        self.daily.set_index('Date').plot(y='Brutte_aval',title = 'Beauharnois', legend=True, ax=self.sc.axes).legend(loc='upper left')
        self.daily.set_index('Date').plot(y='Validee_aval', ax=self.sc.axes).legend(loc='upper left')
        
        toolbar = NavigationToolbar(self.sc, self)
        plotlayout.addWidget(toolbar)
        plotlayout.addWidget(self.sc)
        self.layout.addLayout(plotlayout, row, col)
        return plotlayout
    
    def displayCalendar(self):
        calendar_layout = QHBoxLayout()

        self.dateedit = QDateEdit(calendarPopup=True)
        self.dateedit.setAlignment(Qt.AlignCenter)
        self.dateedit.setFont(QFont('Arial', 20))

        date_str = "2022-10-17 00:00:00"
        qdate = QDateTime.fromString(date_str, "yyyy-M-d hh:mm:ss")

        self.dateedit.setDateTime(qdate)
        self.dateedit.setMaximumDateTime(qdate)
        self.dateedit.setDisplayFormat("dd MMM yyyy")
        self.dateedit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.dateedit.dateTimeChanged.connect(lambda: self.dateUpdated())

        prev_button = QPushButton("<")
        prev_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        prev_button.clicked.connect(lambda:self.prevButtonClicked())
        prev_button.setFont(QFont('Arial', 20))
        next_button = QPushButton(">")
        next_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        next_button.clicked.connect(lambda:self.nextButtonClicked())
        next_button.setFont(QFont('Arial', 20))

        calendar_layout.addWidget(prev_button)
        calendar_layout.addWidget(self.dateedit)
        calendar_layout.addWidget(next_button)

        calendar_layout.setStretch(1, 5)

        return calendar_layout
    
    def displayErrors(self, row, col):
        error_layout = QVBoxLayout()
        # title = QLabel()
        # title.setText("Erreurs:")
        # title.setFont(QFont('Arial', 30))
        # error_layout.addWidget(title)


        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons

        temp_var = self.dateedit.date() 
        date = temp_var.toPyDate()

        no_error_selection = QPushButton(f"Erreurs du jour {date.day}-{date.month}-{date.year}")
        no_error_selection.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        no_error_selection.clicked.connect(partial(self.errorButtonClicked, 0))
        no_error_selection.setFont(QFont('Arial', 20))
        if self.selectedError ==0:
            no_error_selection.setStyleSheet("background-color: gray")
        vbox.addWidget(no_error_selection)

        for i in range(1,50):
            error_selection = QPushButton(f"{i} - 00:00:00")
            error_selection.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            error_selection.clicked.connect(partial(self.errorButtonClicked, i))
            error_selection.setFont(QFont('Arial', 20))
            if i == self.selectedError:
                error_selection.setStyleSheet("background-color: gray")
            vbox.addWidget(error_selection)

        self.widget.setLayout(vbox)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        error_layout.addWidget(self.scroll)
        self.layout.addLayout(error_layout, row , col)

        pass

    def displayMenu(self, row, col):
        menu_layout = QVBoxLayout()

        download_report = QPushButton("Télécharger rapport")
        download_report.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # download_report.clicked.connect()
        download_report.setFont(QFont('Arial', 20))

        change_case = QPushButton("Changer de centrale")
        change_case.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # change_case.clicked.connect()
        change_case.setFont(QFont('Arial', 20))

        menu_layout.addWidget(change_case)
        menu_layout.addWidget(download_report)
        self.layout.addLayout(menu_layout, row, col)

    def displayInformation(self, row, col):
        panel = QWidget() 
        information_layout = QVBoxLayout(panel)
        information_label = QLabel()
        if self.selectedError == 0:
            text = "Information générale"
        else:
            text = f"Information sur erreur {self.selectedError}"
        information_label.setText(text)
        # information_label.setAlignment(Qt.AlignUpperLeft)
        information_layout.addWidget(information_label)
        panel.setStyleSheet("background-color: white;"
                             "border-style: solid;"
                             "border-width: 5px;"
                             "border-color: gray;"
                             "border-radius: 3px")
        information_label.setStyleSheet("border-width: 0px;")
        self.layout.addWidget(panel, row, col)


    def prevButtonClicked(self):
        temp_var = self.dateedit.date()
        temp_var = temp_var.addDays(-1)
        self.dateedit.setDate(temp_var)

    def nextButtonClicked(self):
        temp_var = self.dateedit.date()
        temp_var = temp_var.addDays(1)
        self.dateedit.setDate(temp_var)

    def dateUpdated(self):
        self.selectedError = 0 
        self.update()

    def errorButtonClicked(self, i):
        print(f'pushed button {i}')
        self.selectedError = i

        self.update()

    def update(self):
        self.displayData(1, 0)
        self.displayMenu(2,1)
        self.displayInformation(2, 0)
        self.displayErrors(1,1)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)


    # Subclass QMainWindow to customize your application's main window


    # Create a Qt widget, which will be our window.
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()


    # Your application won't reach here until you exit and the event
    # loop has stopped.
