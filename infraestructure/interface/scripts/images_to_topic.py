import glob
import traceback

# import camera_info_manager
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

class ImageFolderPublisher(object):
    def __init__(self):
        self.folder = rospy.get_param("~dir", "")
        self.topic_name = rospy.get_param("~topic_name", "")
        self.format = rospy.get_param("~format", "")
        # self.encoding = rospy.get_param("~encoding", "bgr8")
        self.frequency = 1.0 / rospy.get_param("~frequency", 15)
        self.loop = rospy.get_param("~loop", False)
        
        # print(self.folder)

        self.get_images()

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(self.topic_name, Image, queue_size=1, latch=True)
        # self.image_pub_depth = rospy.Publisher("depth_images", Image, queue_size=3)

        self.timer = rospy.Timer(rospy.Duration(self.frequency), self.update)

        # s = rospy.Service(self.topic_name+'/publish_images', Trigger, self.trigger)

    def get_images(self):
        self.image_idx = 0
        self.image_list = glob.glob(self.folder + "/*.png")
        # self.image_list_depth = glob.glob(self.folder + "/depth/*.png")
        # self.image_list += glob.glob(self.folder + "/*.jpg")
        self.image_list.sort()
        if len(self.image_list) == 0:
            rospy.logwarn("no images in {}".format(self.folder))
            exit(-1)

    # def trigger(self, req):
    #     self.get_images()
    #     if self.timer == None:
    #         self.timer = rospy.Timer(rospy.Duration(self.frequency), self.update)
    #     return True

    def update(self, event):
        if len(self.image_list) == 0:
            rospy.loginfo("Finished!")

            exit(-1)

        img = self.image_list[0]

        if self.format == "color":
            cv_image = cv2.imread(img, cv2.IMREAD_COLOR)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        elif self.format == "gray":
            cv_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'passthrough')
        else:
            rospy.logwarn("Format not supported. Choose 'color' or 'gray'")
            exit(-1)

        img_msg.header.stamp = event.current_real
            
        self.image_idx +=1

        self.image_pub.publish(img_msg)
        self.image_list.pop(0)
        if self.loop:
            self.image_list.append(img)

if __name__ == '__main__':
    rospy.init_node("ImagesToTopic")
    image_folder_publisher = ImageFolderPublisher()
    rospy.spin()