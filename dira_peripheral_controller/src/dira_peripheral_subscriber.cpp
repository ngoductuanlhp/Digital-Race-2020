#include <termios.h>
#include "iostream"
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "dira_peripheral_controller.h"
#include "geometry_msgs/Point.h"
using namespace std;
bool STATUS1 = false;
bool STATUS2 = false;
bool STATUS3 = false;
bool STATUS4 = false;
bool STATUS5 = false;
bool STATUS6 = false;
ApiHAL HAL;
LCDI2C *lcd = new LCDI2C();
I2C *i2c_device = new I2C();
struct Point {
    int x;
    int y;
};
Point cursor;
ros::Subscriber* sub;
int hexadecimalToDecimal(const char hexVal[]) 
{    
    int len = strlen(hexVal);
    ROS_INFO("%d", len);
      
    // Initializing base value to 1, i.e 16^0 
    int base = 1; 
      
    int dec_val = 0; 
      
    // Extracting characters as digits from last character 
    for (int i=len-1; i>=0; i--) 
    {    
        // if character lies in '0'-'9', converting  
        // it to integral 0-9 by subtracting 48 from 
        // ASCII value. 
        if (hexVal[i]>='0' && hexVal[i]<='9') 
        { 
            dec_val += (hexVal[i] - 48)*base; 
                  
            // incrementing base by power 
            base = base * 16; 
        } 
  
        // if character lies in 'A'-'F' , converting  
        // it to integral 10 - 15 by subtracting 55  
        // from ASCII value 
        else if (hexVal[i]>='A' && hexVal[i]<='F') 
        { 
            dec_val += (hexVal[i] - 55)*base; 
          
            // incrementing base by power 
            base = base*16; 
        }
        else if (hexVal[i]>='a' && hexVal[i]<='f')
        {
            dec_val += (hexVal[i] - 87)*base;

            base = base*16;
        }
    } 
      
    return dec_val; 
} 
void set_led(const std_msgs::Bool::ConstPtr& msg)
{
    STATUS6 = msg->data;
	if (STATUS6)
	{
        HAL.gpioSetValue(LED_PIN, true);
	}
    else
    {
        HAL.gpioSetValue(LED_PIN, false);
    }
}
void lcd_print(const std_msgs::String::ConstPtr& msg){
    std::string str;
    str = msg->data;
    string segment;
    vector<string> array;
    stringstream ss(str);
    while(getline(ss, segment, ':'))
    {
        array.push_back(segment);
    }
    cursor.x = atoi(array.at(0).c_str());
    cursor.y = atoi(array.at(1).c_str());
    int n = array.at(2).length(); 
    char char_array[n+1];
    strcpy(char_array, array.at(2).c_str());
    HAL.setCursor(cursor.x,cursor.y);
    HAL.print(char_array);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "peripheral_subscriber");
    unsigned char lcd_adr;
    ros::NodeHandle n("~");
    ros::NodeHandle nh;
    ros::Publisher button1_status;
    ros::Publisher button2_status;
    ros::Publisher button3_status;
    ros::Publisher button4_status;
    // ros::Publisher sensor1_status;
    ros::Publisher sensor2_status;

    ros::Subscriber sub1 = nh.subscribe("lcd_print", 10, lcd_print);
    ros::Subscriber sub2 = nh.subscribe("led_status", 10, set_led);
    button1_status       = nh.advertise<std_msgs::Bool>("bt1_status",10);
    button2_status       = nh.advertise<std_msgs::Bool>("bt2_status",10);
    button3_status       = nh.advertise<std_msgs::Bool>("bt3_status",10);
    button4_status       = nh.advertise<std_msgs::Bool>("bt4_status",10);
    // sensor1_status       = nh.advertise<std_msgs::Bool>("ss1_status",10);
    sensor2_status       = nh.advertise<std_msgs::Bool>("ss2_status",10);
    
	std_msgs::Bool status1;
    std_msgs::Bool status2;
    std_msgs::Bool status3;
    std_msgs::Bool status4;
    std_msgs::Bool status5;
    std_msgs::Bool status6;

    std::string i2c_add;
    int i2c_add_dec;
    n.param("lcd_i2c_adr", i2c_add, std::string("3f"));
    i2c_add_dec = hexadecimalToDecimal(i2c_add.c_str());
    
    ROS_INFO("Setting I2C address to 0x%s, in decimal: %d",i2c_add.c_str(), i2c_add_dec);
    lcd_adr = (unsigned char) i2c_add_dec;
    i2c_device->m_i2c_bus = 1;
    HAL.initPin(lcd_adr);
    sleep(1);

    unsigned int bt1_status = -1;
    unsigned int bt2_status = -1;
    unsigned int bt3_status = -1;
    unsigned int bt4_status = -1;
    unsigned int ss1_status = -1;
    unsigned int ss2_status = -1;
    
    ros::Rate r(10);
    while (ros::ok()){
        
        HAL.gpioGetValue(SW1_PIN, &bt1_status);
        HAL.gpioGetValue(SW2_PIN, &bt3_status);
        HAL.gpioGetValue(SW3_PIN, &bt2_status);
        HAL.gpioGetValue(SW4_PIN, &bt4_status);
        // HAL.gpioGetValue(SS1_PIN, &ss1_status);
        HAL.gpioGetValue(SS2_PIN, &ss2_status);

        if (bt1_status == HIGH){
            STATUS1 = true;          
        }
        else{
            STATUS1 = false;
        }
        if (bt2_status == HIGH){
            STATUS2 = true;      
        }
        else{
            STATUS2 = false;
        }
        if (bt3_status == HIGH){
            STATUS3 = true;
        }
        else{
            STATUS3 = false;
        }
        if (bt4_status == HIGH){
            STATUS4 = true;
        }
        else{
            STATUS4 = false;
        }
        // if (ss1_status == HIGH){
        //     STATUS5 = true;
        // }
        // else{
        //     STATUS5 = false;
        // }
        if (ss2_status == HIGH){
            STATUS6 = true;
        }
        else{
            STATUS6 = false;
        }
        status1.data = STATUS1;
        button1_status.publish(status1);
        status2.data = STATUS2;
        button2_status.publish(status2);
        status3.data = STATUS3;
        button3_status.publish(status3);
        status4.data = STATUS4;
        button4_status.publish(status4);
        // status5.data = STATUS5;
        // sensor1_status.publish(status5);
        status6.data = STATUS6;
        sensor2_status.publish(status6);
        ros::spinOnce();
        r.sleep();
    }
	return 0;
}

