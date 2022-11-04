
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout, QDateEdit, QHBoxLayout, QSizePolicy, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtCore import QSize, Qt, QDateTime, QDate
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from functools import partial
import pandas as pd
import sys
# from utils import get_df

def get_df(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["Hour"] = df["Date"].dt.hour
    return(df)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        path = 'ui/corrected_df.csv'
        error_path = 'ui/errors.csv'
        self.data = get_df(path)#get_df()
        self.errors = pd.read_csv(error_path)
        self.setWindowTitle("Hack-QC - IUVO-AI")
        self.selectedError = 0
        self.selectedView = 0 
        self.showInformation = {}
        self.layout = QGridLayout()

        calendar_layout = self.displayCalendar()
        self.layout.addLayout(calendar_layout, 0, 1)
        self.displayTitle(0,0)
        self.displayErrors(1,1)
        self.displayData(1, 0)
        self.displayMenu(2,1)
        self.displayInformation(2, 0)

        
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
        title_layout = QHBoxLayout()

        icon = QLabel()
        pixmap = QPixmap('ui/hydro-small.png')
        
        icon.setPixmap(pixmap.scaled(100, 100))
        icon.setStyleSheet(
                        "border-style: solid;"
                        "border-width: 5px;"
                        "border-color: blue;"
                        "border-radius: 3px")

        title = QLabel()
        title.setText("S.C.E.A.U. fait par <font color=teal>IUVO-AI</font>")
        title.setFont(QFont('Arial', 30))
        title.setAutoFillBackground(True)
        title.setStyleSheet("background-color: white;"
                             "border-style: solid;"
                             "border-width: 5px;"
                             "border-color: gray;"
                             "border-radius: 3px")
        title.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(icon)
        title_layout.addWidget(title)
        title_layout.setStretch(1,5)
        self.layout.addLayout(title_layout, row, col)

    def displayData(self, row, col):
        plotlayout = QVBoxLayout()
        temp_var = self.dateedit.date() 
        date = temp_var.toPyDate()
        
        ####### TODO: TO CHANGE LATER MAYBE IF WE WANT TO DO ZOOM 
        info_day = self.data.query(f'Year == {date.year} and Month == {date.month} and Day == {date.day}')
        if self.selectedError == 0:
            self.daily = self.data[info_day.index[0]-36 : info_day.index[-1]]
        elif self.selectedError > 0:
            start = info_day[info_day['Date']==self.error_day.iloc[self.selectedError-1]['Start']].index.values[0]
            finish = info_day[info_day['Date']==self.error_day.iloc[self.selectedError-1]['Finish']].index.values[0]
            self.daily = self.data[start-6 : finish+6]
        
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        # Date Brutte_aval

        if self.selectedView <= 0:
            self.daily.set_index('Date').plot(label= 'Bruttes', y='Brutte_aval', title = 'Beauharnois', legend=True, ax=self.sc.axes).legend(loc='upper left')
        if self.selectedView >= 0:
            self.daily.set_index('Date').plot(label= 'Corrigées', y='Corrected_Brutte_aval', color = 'orange', title = 'Beauharnois', legend=True, ax=self.sc.axes).legend(loc='upper left')
        self.sc.axes.title.set_size(40)
        
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

        temp_var = self.dateedit.date() 
        date = temp_var.toPyDate()
        self.error_day = self.errors.query(f'Year == {date.year} and Month == {date.month} and Day == {date.day}')
        

        for i in range(1,len(self.error_day)+1):
            error_selection = QPushButton(f"{i} - {str(self.error_day.iloc[i-1]['Start'])[-8:]}")
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

        change_case = QPushButton("Changer de plan d'eau")
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
        information_title = QLabel()
        ### TODO: add extra things here -> add choice of different recommendations / manual input
        if self.selectedError == 0:
            text = "Information générale"
            info = f"Nombre d'erreur : {len(self.error_day)} \n"
        else:
            text = f"Information sur erreurs #{self.selectedError}"
            if self.error_day.iloc[self.selectedError-1]['Error'] == 'Random_Spikes':
                text_error = 'pointes aléatoires'
            if self.error_day.iloc[self.selectedError-1]['Error'] == 'Lower_outliers':
                text_error = 'valeurs aberrantes inférieures'
            if self.error_day.iloc[self.selectedError-1]['Error'] == 'Upper_outliers':
                text_error = 'valeurs aberrantes supérieures'

            info = f"Raison de la correction : {text_error} \n" \
                f"Duree : X \n" \
                    f"Moyenne : X \n"\
                        f"Mediane : X \n"
        information_title.setText(text)
        information_title.setFont(QFont('Arial', 20))
        information_label.setText(info)
        # information_label.setAlignment(Qt.AlignUpperLeft)

        information_menu_layout = QHBoxLayout()
        show_label = QLabel()
        show_label.setText('Affichage: ')
        show_label.setAlignment(Qt.AlignRight)
        self.show_all_button = QPushButton("Tout")
        self.show_all_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.show_all_button.setCheckable(True)
        self.show_all_button.clicked.connect(self.showAllButtonClicked)
        self.show_validated_button = QPushButton("Corrigé")
        self.show_validated_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.show_validated_button.setCheckable(True)
        self.show_validated_button.clicked.connect(self.showValidatedButtonClicked)
        self.show_raw_button = QPushButton("Brutte")
        self.show_raw_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.show_raw_button.setCheckable(True)
        self.show_raw_button.clicked.connect(self.showRawButtonClicked)

        if self.selectedView == 0:
            self.show_all_button.setChecked(True)
        elif self.selectedView == 1:
            self.show_validated_button.setChecked(True)
        elif self.selectedView == -1:
            self.show_raw_button.setChecked(True)

        change_label = QLabel()
        change_label.setText('Correction: ')
        change_label.setAlignment(Qt.AlignRight)
        flag_button = QPushButton("Signaler")
        flag_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        manual_button = QPushButton("Manuel")
        manual_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        deactivate_button = QPushButton("Déactiver")
        deactivate_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        information_menu_layout.addWidget(show_label)
        information_menu_layout.addWidget(self.show_all_button)
        information_menu_layout.addWidget(self.show_validated_button)
        information_menu_layout.addWidget(self.show_raw_button)
        information_menu_layout.addWidget(change_label)
        information_menu_layout.addWidget(flag_button)
        information_menu_layout.addWidget(manual_button)
        information_menu_layout.addWidget(deactivate_button)



        information_layout.addWidget(information_title)
        information_layout.addWidget(information_label)
        information_layout.addLayout(information_menu_layout)
        information_layout.setStretch(1, 5)
        panel.setObjectName('panel')
        panel.setStyleSheet("QWidget#panel {background-color: white;"
                             "border-style: solid;"
                             "border-width: 5px;"
                             "border-color: gray;"
                             "border-radius: 3px}")
        # information_label.setStyleSheet("border-width: 0px;")
        self.layout.addWidget(panel, row, col)

    def showAllButtonClicked(self):
        self.show_validated_button.setChecked(False)
        self.show_raw_button.setChecked(False)
        self.selectedView = 0
        self.update()

    def showValidatedButtonClicked(self):
        self.show_all_button.setChecked(False)
        self.show_raw_button.setChecked(False)
        self.selectedView = 1
        self.update()

    def showRawButtonClicked(self):
        self.show_validated_button.setChecked(False)
        self.show_all_button.setChecked(False)
        self.selectedView = -1
        self.update()

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
        self.selectedError = i
        self.update()

    def update(self):
        self.displayData(1, 0)
        self.displayMenu(2,1)
        self.displayErrors(1,1)
        self.displayInformation(2, 0)

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
    # window.setStyleSheet("QMainWindow {background: 'dark grey';}")
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()


    # Your application won't reach here until you exit and the event
    # loop has stopped.
