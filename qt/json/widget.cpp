#include "widget.h"
#include "ui_widget.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <QMessageBox>
#include <QMouseEvent>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    setWindowFlag(Qt::FramelessWindowHint);  //无边框设置

    QJsonObject rootObj;
    rootObj["key"]     = "k";

    //通过QJsonDocument将json数据转为QByteArray
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();

    //存储
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
    QFile file1(jsonFilePath1);
    file1.open(QIODevice::WriteOnly);
    file1.write(jsonArry);
    file1.close();

    process = new QProcess;
    process->start("bash");
    process4 = new QProcess;
    process4->start("bash");

}

Widget::~Widget()
{
    delete ui;
}

void Widget::tinzhi()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::mousePressEvent(QMouseEvent *event)
{
    //鼠标当前位置  event->globalPos()  窗口当前位置this->pos()
    if(event->button() == Qt::LeftButton){
        // qDebug() << event->globalPos() << this->pos();
        mOffset = event->globalPos()-this->pos();
    }
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    //窗口新位置
    this->move(event->globalPos() - mOffset);
}

void Widget::on_pushButton_i_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "i";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_i_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = ",";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_l_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "l";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_l_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_j_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "j";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_j_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_o_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "o";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_o_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton__pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = ".";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton__released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_m_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "m";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_m_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_u_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "u";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_u_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_break_clicked()
{
    if(dipan == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("底盘控制未关闭，请关闭底盘控制！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(jiantu == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图未关闭，请关闭建图！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(twoDdaohang == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("2D导航未关闭，请关闭2D导航！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(zidinwei == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("小车自定位未关闭，请关闭自定位！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hangdian == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("请保存导航点！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hddaohang == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("多点导航未结束，请结束多点导航！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(dipan == "no" && twoDdaohang == "no" && zidinwei == "no" && hangdian == "no" && hddaohang == "no" && jiantu == "no"){
        this->close();
    }
}

void Widget::on_pushButton_k_clicked()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_q_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "q";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_q_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_z_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "z";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_z_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_w_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "w";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_w_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_x_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "x";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_x_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_e_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "e";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_e_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_c_pressed()
{
    QJsonObject rootObj;
    rootObj["key"]     = "c";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_c_released()
{
    QJsonObject rootObj;
    rootObj["key"]     = "k";
    QJsonDocument jsonDoc(rootObj);
    QByteArray jsonArry = jsonDoc.toJson();
    QFile file(jsonFilePath);
    file.open(QIODevice::WriteOnly);
    file.write(jsonArry);
    file.close();
}

void Widget::on_pushButton_2_clicked()
{
    if(dipan == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("底盘已打开，请误重复打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(dipan == "no"){
        // process1 = new QProcess;
        // process1->start("bash");
        // process2 = new QProcess;
        // process2->start("bash");

        // process1->write("cd\n");
        // process1->write(wtr2.toLocal8Bit() + '\n');
        // process1->write("cd\n");
        // process1->write("ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py\n");

        // process2->write("cd\n");
        // process2->write(wtr2.toLocal8Bit() + '\n');
        // process2->write("cd\n");
        // process2->write("ros2 run wheeltec_robot_keyboard wheeltec_keyboard1\n");
        dipan = "ok";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("底盘控制已打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_3_clicked()
{
    if(dipan == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("底盘未打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(dipan == "ok"){
        // process->write("cd\n");
        // process->write("killall static_transform_publisher\n");
        // process->write("killall imu_filter_madgwick_node\n");
        // process->write("killall ekf_node\n");
        // process->write("killall robot_state_publisher\n");
        // process->write("killall joint_state_publisher\n");
        // process->write("killall wheeltec_robot_node\n");
        // process->write("killall wheeltec_keyboard1\n");
        // process->write("killall wheeltec_robot_node\n");
        // process1->terminate();
        // process2->terminate();
        dipan = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("底盘已关闭！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_4_clicked()
{
    // if(rviz2 == "ok"){
    //     QMessageBox mes;
    //     mes.setWindowTitle("提示");
    //     mes.setText("rviz2已打开，请误重复打开！");
    //     mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
    //     mes.setStandardButtons(QMessageBox::Ok);   //ok退出
    //     mes.exec();
    // }
    // if(rviz2 == "no"){
    process4->write("cd\n");
    process4->write(xitong.toLocal8Bit() + '\n');
    process4->write("cd\n");
    process4->write("rviz2\n");
    //     rviz2 = "ok";
    //     QMessageBox mes;
    //     mes.setWindowTitle("提示");
    //     mes.setText("rviz2打开成功！");
    //     mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
    //     mes.setStandardButtons(QMessageBox::Ok);   //ok退出
    //     mes.exec();
    // }
}

// void Widget::on_pushButton_5_clicked()
// {
//     if(rviz2 == "no"){
//         QMessageBox mes;
//         mes.setWindowTitle("提示");
//         mes.setText("rviz2未打开！");
//         mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
//         mes.setStandardButtons(QMessageBox::Ok);   //ok退出
//         mes.exec();
//     }
//     if(rviz2 == "ok"){
//         process->write("cd\n");
//         process->write("killall rviz2\n");
//         rviz2 = "no";
//         QMessageBox mes;
//         mes.setWindowTitle("提示");
//         mes.setText("rviz2已关闭！");
//         mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
//         mes.setStandardButtons(QMessageBox::Ok);   //ok退出
//         mes.exec();
//     }
// }

void Widget::on_pushButton_6_clicked()
{
    if(jiantu == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图已打开，请勿重复打开！可在 小车控制 -> 建图控制 控制小车。");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(jiantu == "no"){
        // process3 = new QProcess;
        // process3->start("bash");
        // process3->write("cd\n");
        // process3->write(wtr2.toLocal8Bit() + '\n');
        // process3->write("cd\n");
        // process3->write("ros2 launch wheeltec_cartographer cartographer.launch.py\n");
        jiantu = "ok";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图已打开！可在 小车控制 -> 建图控制 控制小车。");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_7_clicked()
{
    if(jiantu == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图未打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(jiantu == "ok"){
        // process->write("cd\n");
        // process->write("killall static_transform_publisher\n");
        // process->write("killall imu_filter_madgwick_node\n");
        // process->write("killall ekf_node\n");
        // process->write("killall robot_state_publisher\n");
        // process->write("killall joint_state_publisher\n");
        // process->write("killall cartographer_node\n");
        // process->write("killall cartographer_occupancy_grid_node\n");
        // process->write("killall ros_main\n");
        // process->write("killall wheeltec_robot_node\n");
        // process->write("killall wheeltec_keyboard1\n");
        // process->write("killall wheeltec_robot_node\n");
        // process2->terminate();
        // process3->terminate();
        jiantu = "no";
        jtkz = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图已关闭!");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }

}

void Widget::on_pushButton_8_clicked()
{
    if(jiantu == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图未打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(jiantu == "ok"){
        if(jtkz == "ok"){
            QMessageBox mes;
            mes.setWindowTitle("提示");
            mes.setText("建图控制已打开，请勿重复打开！");
            mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
            mes.setStandardButtons(QMessageBox::Ok);   //ok退出
            mes.exec();
        }
        if(jtkz == "no"){
            // process2 = new QProcess;
            // process2->start("bash");
            // process2->write("cd\n");
            // process2->write(wtr2.toLocal8Bit() + '\n');
            // process2->write("cd\n");
            // process2->write("ros2 run wheeltec_robot_keyboard wheeltec_keyboard1\n");
            jtkz = "ok";
            QMessageBox mes;
            mes.setWindowTitle("提示");
            mes.setText("建图控制已打开，可通过右面按钮控制。");
            mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
            mes.setStandardButtons(QMessageBox::Ok);   //ok退出
            mes.exec();
        }
    }
}

void Widget::on_pushButton_9_clicked()
{
    if(jiantu == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("建图未打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(jiantu == "ok"){
        // process->write("cd\n");
        // process->write(wtr2.toLocal8Bit() + '\n');
        // process->write("cd\n");
        // process->write("ros2 launch wheeltec_nav2 save_map.launch.py\n");
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("地图已保存！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_10_clicked()
{
    if(twoDdaohang == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("2D导航已打开，请勿重复打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(twoDdaohang == "no"){
        // process3 = new QProcess;
        // process3->start("bash");
        // process3->write("cd\n");
        // process3->write(wtr2.toLocal8Bit() + '\n');
        // process3->write("cd\n");
        // process3->write("ros2 launch wheeltec_nav2 wheeltec_nav2.launch.py\n");
        twoDdaohang = "ok";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("2D导航已打开!");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_11_clicked()
{
    if(twoDdaohang == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("2D导航未打开！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(twoDdaohang == "ok"){
        // process->write("cd\n");
        // process->write("killall static_transform_publisher\n");
        // process->write("killall imu_filter_madgwick_node\n");
        // process->write("killall ekf_node\n");
        // process->write("killall robot_state_publisher\n");
        // process->write("killall joint_state_publisher\n");
        // process->write("killall ros_main\n");
        // process->write("killall amcl\n");
        // process->write("killall bt_navigator\n");
        // process->write("killall behavior_server\n");
        // process->write("killall controller_server\n");
        // process->write("killall map_server\n");
        // process->write("killall planner_server\n");
        // process->write("killall smoother_server\n");
        // process->write("killall waypoint_follower\n");
        // process->write("killall lifecycle_manager\n");
        // process->write("killall wheeltec_robot_node\n");
        // process3->terminate();
        twoDdaohang = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("2D导航已关闭！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}


void Widget::on_pushButton_12_clicked()
{
    if(zidinwei == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("自定位已开启,请勿重复开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(zidinwei == "no"){
        // process5 = new QProcess;
        // process5->start("bash");
        // process5->write("cd\n");
        // process5->write(mir2.toLocal8Bit() + '\n');
        // process5->write("cd\n");
        // process5->write("ros2 run snap_map_icp snap_map\n");
        zidinwei = "ok";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("自定位已开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_13_clicked()
{
    if(zidinwei == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("自定位未开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(zidinwei == "ok"){
        process->write("cd\n");
        process->write("killall snap_map\n");
        process5->terminate();
        zidinwei = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("已关闭自定位！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_14_clicked()
{
    if(hangdian == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("设置航点已开启，请勿重复开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hangdian == "no"){
        // process3 = new QProcess;
        // process3->start("bash");
        // process3->write("cd\n");
        // process3->write(wpr2.toLocal8Bit() + '\n');
        // process3->write("cd\n");
        // process3->write("ros2 launch wp_map_tools add_waypoint_sim.launch.py\n");
        hangdian = "ok";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("设置航点已开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}


void Widget::on_pushButton_15_clicked()
{
    if(hangdian == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("设置航点未开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hangdian == "ok"){
        // process->write("cd\n");
        // process->write(wpr2.toLocal8Bit() + '\n');
        // process->write("ros2 run wp_map_tools wp_saver\n");
        // process3->terminate();
        hangdian = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("航点已保存，已关闭航点设置！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}


void Widget::on_pushButton_16_clicked()
{
    if(hddaohang == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("航点导航未开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hddaohang == "ok"){
        QJsonObject rootObj;
        rootObj["key"]     = "p";
        QJsonDocument jsonDoc(rootObj);
        QByteArray jsonArry = jsonDoc.toJson();
        QFile file(jsonFilePath1);
        file.open(QIODevice::WriteOnly);
        file.write(jsonArry);
        file.close();
        hddaohang = "no";
        ksdh = "no";
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("航点导航已关闭！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}


void Widget::on_pushButton_17_clicked()
{
    if(hddaohang == "ok"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("航点导航已开启，请勿重复开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hddaohang == "no"){
        hddaohang = "ok";
        // process6 = new QProcess;
        // process6->start("bash");
        // process6->write("cd\n");
        // process6->write(wpr2.toLocal8Bit() + '\n');
        // process6->write("cd\n");
        // process6->write("ros2 launch wp_map_tools marker_navigation1.launch.py\n");
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("航点导航已开启！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
}

void Widget::on_pushButton_18_clicked()
{
    if(hddaohang == "no"){
        QMessageBox mes;
        mes.setWindowTitle("提示");
        mes.setText("未启动导航！");
        mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
        mes.setStandardButtons(QMessageBox::Ok);   //ok退出
        mes.exec();
    }
    if(hddaohang == "ok"){
        if(ksdh == "ok"){
            QMessageBox mes;
            mes.setWindowTitle("提示");
            mes.setText("已开始导航，请勿重复开始！");
            mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
            mes.setStandardButtons(QMessageBox::Ok);   //ok退出
            mes.exec();
        }
        if(ksdh == "no"){
            // process2 = new QProcess;
            // process2->start("bash");
            // process2->write("cd\n");
            // process2->write(wtr2.toLocal8Bit() + '\n');
            // process2->write("cd\n");
            // process2->write("ros2 run wheeltec_robot_keyboard wheeltec_keyboard1\n");
            ksdh = "ok";
            QMessageBox mes;
            mes.setWindowTitle("提示");
            mes.setText("航点导航已开始！");
            mes.setStyleSheet("QPushButton {color:black}");   //字体颜色
            mes.setStandardButtons(QMessageBox::Ok);   //ok退出
            mes.exec();
        }
    }
}

