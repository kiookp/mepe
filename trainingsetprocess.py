import poseembedding as pe                      # 姿态关键点编码模块
import poseclassifier as pc                     # 姿态分类器
import extracttrainingsetkeypoints as ek        # 提取训练集关键点特征
import csv
import os



def trainset_process(flag):
    # 如果fitness_poses_csvs_out文件夹下的push_down.csv和push_up.csv已经存在，则不用导入样本图片再训练了
    if flag == 1:
        if os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\push_up.csv')) and os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\push_down.csv')):
            return
    # 如果fitness_poses_csvs_out文件夹下的squat_down.csv和squat_up.csv已经存在，则不用导入样本图片再训练了
    elif flag == 2:
        if os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\squat_up.csv')) and os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\squat_down.csv')):
            return
    # 如果fitness_poses_csvs_out文件夹下的pull_down.csv和pull_up.csv已经存在，则不用导入样本图片再训练了
    elif flag == 3:
        if os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\pull_up.csv')) and os.path.isfile(os.path.join(os.path.dirname(__file__),'fitness_poses_csvs_out\pull_down.csv')):
            return

    # 指定样本图片的路径
    bootstrap_images_in_folder = 'fitness_poses_images_in'

    # Output folders for bootstrapped images and CSVs.
    bootstrap_images_out_folder = 'fitness_poses_images_out'
    bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

    # Initialize helper.
    bootstrap_helper = ek.BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    # Check how many pose classes and images for them are available.
    bootstrap_helper.print_images_in_statistics()

    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses were still saved in
    # the folderd (but not in the CSVs) for debug purpose. Let's remove them.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # Please manually verify predictions and remove samples (images) that has wrong pose prediction. Check as if you were asked to classify pose just from predicted landmarks. If you can't - remove it.
    # Align CSVs and image folders once you are done.

    # Align CSVs with filtered images.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()


    # Transforms pose landmarks into embedding.
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Classifies give pose against database of poses.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))

    # Analyze outliers.
    bootstrap_helper.analyze_outliers(outliers)

    # Remove all outliers (if you don't want to manually pick).
    bootstrap_helper.remove_outliers(outliers)

    # Align CSVs with images after removing outliers.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

