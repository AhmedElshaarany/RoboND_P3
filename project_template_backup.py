#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Define Global parameters
USING_HSV = True

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    #print(type(yaml_dict["test_scene_num"]))
    yaml_dict["arm_name"]  = arm_name.data
    #print(type(yaml_dict["arm_name"]))
    yaml_dict["object_name"] = str(object_name.data)
    #print(type(yaml_dict["object_name"]))
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    #print(type(yaml_dict["pick_pose"]))
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    #print(type(yaml_dict["place_pose"]))
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(15)
    fil.set_std_dev_mul_thresh(0.1)
    filtered_cloud = fil.filter()
    
    # TODO: Voxel Grid Downsampling
    # Create a VoxelGrid filter object for our input point cloud
    vox = filtered_cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.01

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()


    
    # TODO: PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    #axis_min = 0.76
    #axis_max = 1.1
    axis_min = 0.6
    #axis_max = 1.5
    axis_max = 0.77
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()


    
    # TODO: RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)


    
    # TODO: Extract inliers and outliers
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)

    # Extract outliers
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)


    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    #ec.set_ClusterTolerance(0.001)
    #ec.set_MinClusterSize(10)
    #ec.set_MaxClusterSize(250)
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(3000)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)


    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table   = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)


    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)

        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=USING_HSV)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)


    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)




    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()

    try:
        #pr2_mover(detected_objects_list)
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

    
# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()
    object_name = String()
    object_group = String()
    object_arm = String()
    pick_pose = Pose()
    place_pose = Pose()
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # Define scene
    test_scene_num.data = 1

    # TODO: Parse parameters into individual variables
    object_name_list = [object_list_param[i]['name'] for i in range(len(object_list_param))]
    object_group_list = [object_list_param[i]['group'] for i in range(len(object_list_param))]
    object_arm_list = ['left' if(object_list_param[i]['group'] == 'red') else 'right' for i in range(len(object_list_param))]

    dropbox_name_list = [dropbox_list_param[i]['name'] for i in range(len(dropbox_list_param))]
    dropbox_group_list = [dropbox_list_param[i]['group'] for i in range(len(dropbox_list_param))]
    dropbox_position_list = [dropbox_list_param[i]['position'] for i in range(len(dropbox_list_param))]



    
    # TODO: Rotate PR2 in place to capture side tables for the collision map
    #print(object_list_param)
    #print()
    #print(object_name.data)
    #print()
    
    # TODO: Loop through the pick list
    for object in object_list:
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        label = object.label
        #labels.append(label)
        points_arr = ros_to_pcl(object.cloud).to_array()

        object_index = object_name_list.index(label)
        object_name.data = label
        object_arm.data = object_arm_list[object_index]

        # calculate centriods as floats by casting
        # centroids = np.asscalar(np.mean(points_arr, axis=0)[:3])
        centroid = np.mean(points_arr, axis=0)[:3]
        #print(centroid)

        # assign pick_pose for object
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])
        
        # TODO: Create 'place_pose' for the object
        dropbox_index = dropbox_name_list.index(object_arm_list[object_index])
        place_pose.position.x = dropbox_position_list[dropbox_index][0]
        place_pose.position.y = dropbox_position_list[dropbox_index][1]
        place_pose.position.z = dropbox_position_list[dropbox_index][2]

        # TODO: Assign the arm to be used for pick_place
        which_arm = object_arm.data[object_index]
        

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            TEST_SCENE_NUM = test_scene_num
            OBJECT_NAME = object_name
            WHICH_ARM = object_arm
            PICK_POSE = pick_pose
            PLACE_POSE = place_pose
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME , PICK_POSE, PLACE_POSE)
            dict_list.append(yaml_dict)
            print yaml_dict


            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_1.yaml', dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)


    # TODO: Load Model From disk
    #model = pickle.load(open('model_test1_world_400_instances.sav', 'rb'))
    model = pickle.load(open('model_test1_world_200_instances.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
        
