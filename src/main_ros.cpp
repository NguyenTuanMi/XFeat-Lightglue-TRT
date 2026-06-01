#include <rclcpp/rclcpp.hpp>
#include "localizer_node.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<xfeat_localizer::LocalizerNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}