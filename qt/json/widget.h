#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QKeyEvent>
#include <QProcess>
#include <QString>

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:

    void on_pushButton_i_pressed();

    void on_pushButton_i_released();

    void on_pushButton_pressed();

    void on_pushButton_released();

    void on_pushButton_l_pressed();

    void on_pushButton_l_released();

    void on_pushButton_j_pressed();

    void on_pushButton_j_released();

    void on_pushButton_o_pressed();

    void on_pushButton_o_released();

    void on_pushButton__pressed();

    void on_pushButton__released();

    void on_pushButton_m_pressed();

    void on_pushButton_m_released();

    void on_pushButton_u_pressed();

    void on_pushButton_u_released();

    void on_pushButton_break_clicked();

    void on_pushButton_k_clicked();

    void on_pushButton_q_pressed();

    void on_pushButton_q_released();

    void on_pushButton_z_pressed();

    void on_pushButton_z_released();

    void on_pushButton_w_pressed();

    void on_pushButton_w_released();

    void on_pushButton_x_pressed();

    void on_pushButton_x_released();

    void on_pushButton_e_pressed();

    void on_pushButton_e_released();

    void on_pushButton_c_pressed();

    void on_pushButton_c_released();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    // void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_7_clicked();

    void on_pushButton_8_clicked();

    void on_pushButton_9_clicked();

    void on_pushButton_10_clicked();

    void on_pushButton_11_clicked();

    void on_pushButton_12_clicked();

    void on_pushButton_13_clicked();

    void on_pushButton_14_clicked();

    void on_pushButton_15_clicked();

    void on_pushButton_16_clicked();

    void on_pushButton_17_clicked();

    void on_pushButton_18_clicked();

protected:
    void tinzhi();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event) override;

private:
    Ui::Widget *ui;
    QProcess *process;
    QProcess *process1;
    QProcess *process2;
    QProcess *process3;
    QProcess *process4;
    QProcess *process5;
    QProcess *process6;

    QPoint mOffset;

    QString dipan = "no";
    // QString rviz2 = "no";
    QString jiantu = "no";
    QString jtkz = "no";
    QString twoDdaohang = "no";
    QString zidinwei = "no";
    QString hangdian = "no";
    QString hddaohang = "no";
    QString ksdh = "no";

    QString jsonFilePath = "/home/hngxy/Downloads/zz/json/test.json";
    QString jsonFilePath1 = "/home/hngxy/Downloads/zz/json/dipan.json";
    QString xitong = "source /opt/ros/humble/setup.bash";
    QString wtr2 = "source /home/hngxy/wheeltec_ros2/install/setup.bash";
    QString mir2 = "source /home/bxx/map_icp_ros2/install/setup.bash";
    QString wpr2 = "source /home/hngxy/way_point/install/setup.bash";

};
#endif // WIDGET_H
