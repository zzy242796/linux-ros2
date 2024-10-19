/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QHBoxLayout *horizontalLayout_4;
    QTabWidget *tabWidget;
    QWidget *tab;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton_u;
    QPushButton *pushButton_i;
    QPushButton *pushButton_o;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *pushButton_j;
    QPushButton *pushButton_k;
    QPushButton *pushButton_l;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *pushButton_m;
    QPushButton *pushButton;
    QPushButton *pushButton_;
    QPushButton *pushButton_break;
    QPushButton *pushButton_3;
    QWidget *layoutWidget1;
    QGridLayout *gridLayout;
    QPushButton *pushButton_q;
    QPushButton *pushButton_z;
    QPushButton *pushButton_w;
    QPushButton *pushButton_x;
    QPushButton *pushButton_e;
    QPushButton *pushButton_c;
    QPushButton *pushButton_2;
    QWidget *tab_2;
    QWidget *tab_3;
    QWidget *tab_4;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QString::fromUtf8("Widget"));
        Widget->resize(569, 364);
        horizontalLayout_4 = new QHBoxLayout(Widget);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        tabWidget = new QTabWidget(Widget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        layoutWidget = new QWidget(tab);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(120, 140, 256, 171));
        verticalLayout = new QVBoxLayout(layoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        pushButton_u = new QPushButton(layoutWidget);
        pushButton_u->setObjectName(QString::fromUtf8("pushButton_u"));

        horizontalLayout->addWidget(pushButton_u);

        pushButton_i = new QPushButton(layoutWidget);
        pushButton_i->setObjectName(QString::fromUtf8("pushButton_i"));

        horizontalLayout->addWidget(pushButton_i);

        pushButton_o = new QPushButton(layoutWidget);
        pushButton_o->setObjectName(QString::fromUtf8("pushButton_o"));

        horizontalLayout->addWidget(pushButton_o);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        pushButton_j = new QPushButton(layoutWidget);
        pushButton_j->setObjectName(QString::fromUtf8("pushButton_j"));

        horizontalLayout_2->addWidget(pushButton_j);

        pushButton_k = new QPushButton(layoutWidget);
        pushButton_k->setObjectName(QString::fromUtf8("pushButton_k"));

        horizontalLayout_2->addWidget(pushButton_k);

        pushButton_l = new QPushButton(layoutWidget);
        pushButton_l->setObjectName(QString::fromUtf8("pushButton_l"));

        horizontalLayout_2->addWidget(pushButton_l);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        pushButton_m = new QPushButton(layoutWidget);
        pushButton_m->setObjectName(QString::fromUtf8("pushButton_m"));

        horizontalLayout_3->addWidget(pushButton_m);

        pushButton = new QPushButton(layoutWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout_3->addWidget(pushButton);

        pushButton_ = new QPushButton(layoutWidget);
        pushButton_->setObjectName(QString::fromUtf8("pushButton_"));

        horizontalLayout_3->addWidget(pushButton_);


        verticalLayout->addLayout(horizontalLayout_3);

        pushButton_break = new QPushButton(tab);
        pushButton_break->setObjectName(QString::fromUtf8("pushButton_break"));
        pushButton_break->setGeometry(QRect(430, 280, 89, 25));
        pushButton_3 = new QPushButton(tab);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setGeometry(QRect(430, 210, 89, 25));
        layoutWidget1 = new QWidget(tab);
        layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(160, 20, 174, 104));
        gridLayout = new QGridLayout(layoutWidget1);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        pushButton_q = new QPushButton(layoutWidget1);
        pushButton_q->setObjectName(QString::fromUtf8("pushButton_q"));

        gridLayout->addWidget(pushButton_q, 0, 0, 1, 1);

        pushButton_z = new QPushButton(layoutWidget1);
        pushButton_z->setObjectName(QString::fromUtf8("pushButton_z"));

        gridLayout->addWidget(pushButton_z, 0, 1, 1, 1);

        pushButton_w = new QPushButton(layoutWidget1);
        pushButton_w->setObjectName(QString::fromUtf8("pushButton_w"));

        gridLayout->addWidget(pushButton_w, 1, 0, 1, 1);

        pushButton_x = new QPushButton(layoutWidget1);
        pushButton_x->setObjectName(QString::fromUtf8("pushButton_x"));

        gridLayout->addWidget(pushButton_x, 1, 1, 1, 1);

        pushButton_e = new QPushButton(layoutWidget1);
        pushButton_e->setObjectName(QString::fromUtf8("pushButton_e"));

        gridLayout->addWidget(pushButton_e, 2, 0, 1, 1);

        pushButton_c = new QPushButton(layoutWidget1);
        pushButton_c->setObjectName(QString::fromUtf8("pushButton_c"));

        gridLayout->addWidget(pushButton_c, 2, 1, 1, 1);

        pushButton_2 = new QPushButton(tab);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setGeometry(QRect(30, 60, 89, 25));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        tabWidget->addTab(tab_4, QString());

        horizontalLayout_4->addWidget(tabWidget);


        retranslateUi(Widget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "Widget", nullptr));
        pushButton_u->setText(QString());
        pushButton_i->setText(QCoreApplication::translate("Widget", "^", nullptr));
        pushButton_o->setText(QString());
        pushButton_j->setText(QCoreApplication::translate("Widget", "<", nullptr));
        pushButton_k->setText(QCoreApplication::translate("Widget", "k", nullptr));
        pushButton_l->setText(QCoreApplication::translate("Widget", ">", nullptr));
        pushButton_m->setText(QString());
        pushButton->setText(QCoreApplication::translate("Widget", "|", nullptr));
        pushButton_->setText(QString());
        pushButton_break->setText(QCoreApplication::translate("Widget", "\351\200\200\345\207\272", nullptr));
        pushButton_3->setText(QCoreApplication::translate("Widget", "\345\205\263\351\227\255\346\216\247\345\210\266", nullptr));
        pushButton_q->setText(QCoreApplication::translate("Widget", "\345\212\240\346\234\200\345\244\247\351\200\237\345\272\246", nullptr));
        pushButton_z->setText(QCoreApplication::translate("Widget", "\345\207\217\346\234\200\345\244\247\351\200\237\345\272\246", nullptr));
        pushButton_w->setText(QCoreApplication::translate("Widget", "\345\212\240\347\272\277\351\200\237\345\272\246", nullptr));
        pushButton_x->setText(QCoreApplication::translate("Widget", "\345\207\217\347\272\277\351\200\237\345\272\246", nullptr));
        pushButton_e->setText(QCoreApplication::translate("Widget", "\345\212\240\350\247\222\351\200\237\345\272\246", nullptr));
        pushButton_c->setText(QCoreApplication::translate("Widget", "\345\207\217\350\247\222\351\200\237\345\272\246", nullptr));
        pushButton_2->setText(QCoreApplication::translate("Widget", "\345\274\200\345\247\213\346\216\247\345\210\266", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QCoreApplication::translate("Widget", "\345\260\217\350\275\246\346\216\247\345\210\266", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QCoreApplication::translate("Widget", "\345\273\272\345\233\276", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QCoreApplication::translate("Widget", "\345\257\274\350\210\252", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QCoreApplication::translate("Widget", "\351\241\265", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
