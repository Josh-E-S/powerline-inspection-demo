\DeclareAcronym
insplad short=InsPLAD, long=Inspection Power Line Asset Dataset, \DeclareAcronymap short=AP, long=Average Precision, \DeclareAcronymauroc short=AUROC, long=Area Under the Receiver Operating Characteristic, \DeclareAcronymuav short=UAV, long=Unmanned Aerial Vehicle, \DeclareAcronymcv short=CV, long=Computer Vision, \DeclareAcronymod short=OD, long=Object Detection, \DeclareAcronymcplid short=CPLID, long=Chinese Power Line Insulator Dataset, \DeclareAcronymss short=SS, long=Semantic Segmentation, \DeclareAcronymis short=IS, long=Instance Segmentation, \DeclareAcronymttpla short=TTPLA, long=Transmission Towers and Power Lines Aerial-Image Dataset, \DeclareAcronymstnplad short=STN PLAD, long=STN Power Line Asset Dataset, \DeclareAcronymic short=IC, long=Image Classification, \DeclareAcronymad short=AD, long=Anomaly Detection, \DeclareAcronymyolo short=YOLO, long=You Only Look Once, \DeclareAcronymrcnn short=R-CNN, long=Region-Based Convolutional Neural Networks, \DeclareAcronymgan short=GAN, long=Generative Adversarial Networks, \DeclareAcronymfcn short=FCN, long=Fully Connected Networks, \DeclareAcronymmlp short=MLP, long=Multilayer Perceptrons, \DeclareAcronymtood short=TOOD, long=Task-aligned One-stage Object Detection, \DeclareAcronymvgg short=VGG, long=Very Deep Convolutional Networks, \DeclareAcronymcoco short=COCO, long=Microsoft Common Objects in Context, \DeclareAcronymiou short=IoU, long=Intersection over Union, \DeclareAcronymssd short=SSD, long=Single Shot MultiBox Detector, \DeclareAcronymognet short=OGNet, long=Old is Gold Net, \DeclareAcronymcsflow short=CS-Flow, long=Fully Convolutional Cross-Scale-Flows, \DeclareAcronymaebad short=AeBAD, long=Aero-engine Blade Anomaly Detection Dataset,

InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images
\nameAndré Luiz Buarque Vieira e Silvaa, Heitor de Castro Felixa, Franscisco Paulo Magalhães Simõesa, d, Veronica Teichrieba, Michel dos Santosb, Hemir Santiagob, Virginia Sgottib and Henrique Lott Netoc
CONTACT A. L. B. Vieira e Silva. Email: albvs@cin.ufpe.braVoxar Labs, Centro de Informática, Universidade Federal de Pernambuco, Recife, Brazil; bIn Forma Software, Recife, Brazil; cSistema de Transmissão Nordeste, Recife, Brazil; dVisual Computing Lab, Departamento de Computação, Universidade Federal Rural de Pernambuco, Recife, Brazil
Abstract
Power line maintenance and inspection are essential to avoid power supply interruptions, reducing its high social and financial impacts yearly. Automating power line visual inspections remains a relevant open problem for the industry due to the lack of public real-world datasets of power line components and their various defects to foster new research. This paper introduces InsPLAD, a Power Line Asset Inspection Dataset and Benchmark containing 10,607 high-resolution Unmanned Aerial Vehicles colour images. The dataset contains seventeen unique power line assets captured from real-world operating power lines. Additionally, five of those assets present six defects: four of which are corrosion, one is a broken component, and one is a bird’s nest presence. All assets were labelled according to their condition, whether normal or the defect name found on an image level. We thoroughly evaluate state-of-the-art and popular methods for three image-level computer vision tasks covered by InsPLAD: object detection, through the AP metric; defect classification, through Balanced Accuracy; and anomaly detection, through the AUROC metric. InsPLAD offers various vision challenges from uncontrolled environments, such as multi-scale objects, multi-size class instances, multiple objects per image, intra-class variation, cluttered background, distinct point-of-views, perspective distortion, occlusion, and varied lighting conditions. To the best of our knowledge, InsPLAD is the first large real-world dataset and benchmark for power line asset inspection with multiple components and defects for various computer vision tasks, with a potential impact to improve state-of-the-art methods in the field. It will be publicly available in its integrity on a repository with a thorough description. It can be found at https://github.com/andreluizbvs/InsPLAD/ or https://drive.google.com/drive/folders/1psHiRyl7501YolnCcB8k55rTuAUcR9Ak?usp=sharing.

keywords: Power line inspection; Datasets and evaluation; Object detection; Image classification; Anomaly detection; Deep learning
List of Abbreviations
AD Anomaly Detection
AeBAD Aero-engine Blade Anomaly Detection Dataset
AUROC Area Under the Receiver Operating Characteristic
TPR True Positive Rate
FPR False Positive Rate
COCO Microsoft Common Objects in Context
CPLID Chinese Power Line Insulator Dataset
CV Computer Vision
FCN Fully Connected Networks
GAN Generative Adversarial Networks
IC Image Classification
InsPLAD Inspection Power Line Asset Dataset
IoU Intersection over Union
IS Instance Segmentation
MLP Multilayer Perceptrons
OD Object Detection
SS Semantic Segmentation
STN PLAD STN Power Line Asset Dataset
TTPLA Transmission Towers and Power Lines Aerial-Image Dataset
UAV Unmanned Aerial Vehicle

1Introduction
Nowadays, power line maintenance and inspection are essential for guaranteeing proper functioning and uninterrupted power supply for human activities. Besides that, blackouts may result in severe social impacts and financial problems for the power supply companies Bruch et al. (2011); Li et al. (2020); Nguyen et al. (2018). Moreover, many issues can arise in power line components, i.e., power line assets, due to their continuous use, mainly because they are often inserted in open, uncontrolled environments. Nature and human factors such as strong winds, rains and storms, intense sun, sea air, pollution, high vegetation, and birds, in general, can cause rust, cracks, gaps, short circuits, and even the breakdown of the power line components.

Maintaining regular power line inspections is fundamental. However, they are costly, dangerous, and time-consuming Rahmani et al. (2013); Hu and Liu (2017). Due to the recent advances in computer vision, automating the inspection process is an attractive alternative since the requirement for human inspectors to climb the transmission towers or analyse massive amounts of inspection images would decrease. In short, automating the inspection process can bring safety and reduce financial costs while providing time gains and increasing inspection frequency.

The importance of public datasets in advancing the state-of-the-art of multiple tasks and applications became clear after their introduction. Some examples are \accoco Lin et al. (2014a), ImageNet Deng et al. (2009) and, for anomaly detection, the MVTec AD Bergmann et al. (2019) and \acaebad Zhang et al. (2023).

Power line inspection works are usually financed by companies and government agencies which choose not to disclose their datasets Nguyen et al. (2018); Siddiqui et al. (2018); Zhang et al. (2019); Yang et al. (2019); Lei and Sui (2019). This may be to maintain some competitive advantage or other confidentiality issues, which, ultimately, hinders the development of this research topic Nguyen et al. (2018); Liu et al. (2020); Xia et al. (2022).

Usually, the inspection process of power lines undergoes multiple steps. A straightforward pipeline may be: 1) Detect power line assets in \acuav inspection images and 2) Classify the assets regarding their conditions, e.g., normal or presenting a known defect. However, defects may present themselves in unlimited shapes and sizes, as well as the absence of asset parts. Therefore, it is only reasonable to catalogue common types of defects. A promising way to deal with the less frequent and even unprecedented defects in the industry is to approach this problem as an unsupervised anomaly detection task, in which defects are treated as anomalies Liu et al. (2023).

We propose the \acinsplad, a new dataset for inspecting power line assets to address multiple research gaps in the field. The data come from real-world inspections and consists of 10,607 Full HD RGB images of 17 unique power line asset categories captured in the wild by a UAV under multiple environmental conditions, orientations, and distances (28,933 asset instances in total). This amount and variability of data allow the evaluation of state-of-the-art object detection methods. Also, for five of those assets, there are six different types of defects annotated on an image level (402 defect samples in total, cropped from the UAV images), allowing the evaluation of image classification and unsupervised anomaly detection methods. Thus, we develop a benchmark to serve as a baseline for the community. Their performance, as well as throughput and model size, are assessed. The evaluations show room for improvement in all tasks proposed by InsPLAD. Also, components of power lines from different companies share high similarities, as verified by the datasets cited in the following section. Considering that methods tested in InsPLAD should also perform similarly in power line assets from multiple companies.

To the best of our knowledge, the InsPLAD is the first public dataset for power line asset inspection that offers tasks for each step in a usual inspection pipeline of power line assets: object detection for detecting power line components in UAV images, defect classification in the cropped power line component images, and unsupervised anomaly detection also in cropped asset images. In addition, InsPLAD can be expanded to other \accv sub-tasks, such as fine-grained classification and small object detection, as it has many images with multiple assets per image. It is intended to spark future research in developing methods for inspection in multiple industries where some defects/faults are not acceptable due to costs or safety concerns, leading to scarcity of data with faulty/anomalous objects, e.g., energy distribution, dangerous spaces, and modern logistics supply chain. In summary, the research gaps from automatic power line visual inspection and the main contributions of InsPLAD to solve them are listed side-by-side in Table 1.

Table 1:Research gap/Contribution pairs addressed by InsPLAD
Research gap
 	
Contribution
Lack of public datasets for power line inspection Nguyen
et al. (2018) Liu
et al. (2020) Xia
et al. (2022).
 	
A novel dataset for power line assets inspection in the wild, using real-world inspection UAV images.
Public power line datasets usually do not address multiple vision tasks Tao
et al. (2020) Tomaszewski et al. (2018) Lee
et al. (2017); Bian
et al. (2019).
 	
Our dataset can be applied in three vision tasks with deep learning methods: object detection for asset detection and image classification/anomaly detection for defect classification. It also offers multiple computer vision challenges with far more images/defects than other datasets.
Power line components present multiple challenging appearances Vieira-e-Silva et al. (2021).
 	
InsPLAD presents several categories of power line assets with multiple subtypes, orientations, and scales in uncontrolled environments (in the wild).
Lack of faulty industrial component imagery taken from real-world scenarios due to costs and safety issues Bergmann et al. (2019).
 	
The first public dataset that covers multiple faulty power line assets, allowing the development of supervised and unsupervised methods.
Lack of a power line inspection benchmark to spark new research and select the best setup for real-world scenarios Vieira-e-Silva et al. (2021) Abdelfattah
et al. (2020).
 	
A comprehensive benchmark with state-of-the-art and popular methods on the proposed dataset to serve as a baseline for future research.
In the following section, we present the main public power line datasets available divided by component types and the main methods applied to those. In Section 3 we describe the proposed dataset, showing the data collection protocol, annotation process and its properties. Next, the benchmark is introduced, divided into three parts: Assets Detection, Supervised Fault Classification, and Unsupervised Anomaly Detection. In each one, we show the experiment setup, results, and discussion. Then, InsPLAD’s strengths and limitations are presented, followed by the conclusions and future works.

2Related Works
Despite the multiple general-purpose datasets available for object detection Lin et al. (2014a); Deng et al. (2009); Everingham et al. (2010); Kuznetsova et al. (2020), image classification Deng et al. (2009); Everingham et al. (2010), and anomaly detection Bergmann et al. (2019), power line inspection condense numerous significant problems for the computer vision community.

The existing public power line datasets only cover some of the various power line components and their defects. They do not have enough data or variability to feed the increasingly data-hungry state-of-the-art methods. Most focus on one or two categories of power line components, mainly Insulators, Transmission Towers, or the conductor itself. In this scenario, InsPLAD presents challenges as multi-scale components, multiple objects per image, and classes with multiple instances of the same component, e.g., Stockbridge Dampers, or underrepresented classes with few samples, e.g., Spacers. It also presents contextual challenges such as cluttered background, uncontrolled outdoor environment, perspective distortion, occlusion, and multiple lighting conditions. Next, the current public power line datasets and the main visual inspection methods for their problems are briefly described.

2.1Datasets
Here, we select the current power line datasets with the following conditions: publicly available data, real-world data, and only data from high-voltage overhead lines. An extensive list of power line datasets is discussed in Ruszczak et al. (2023). Table 2 shows a summarised comparison between the selected datasets and our proposed InsPLAD.

2.1.1Insulator Datasets
Tomaszewski et al. Tomaszewski et al. (2018) propose an \acod Insulator dataset containing 2630 images of size 5616
×
3744, with one Insulator per image. Despite many samples, the images are repetitive since they come from subsequent frames of nine videos made with a still camera pointing to an Insulator piece suspended by a metal structure. In addition, lighting conditions and the Insulator orientation vary between sets of images. The other Insulator dataset is \accplid Tao et al. (2020), which is also an OD dataset that consists of 600 real-world images from power line towers and 248 images of segmented, defective Insulators pasted over different backgrounds. The defective Insulators are missing one or more Insulator caps, and the images are of size 1152
×
864. In it, there are 1,569 annotations, ranging from Insulators and missing caps.

2.1.2Conductor Dataset
The Powerline dataset Lee et al. (2017) is a \acss dataset consisting of 4,200 regular RGB images and 4,200 Infrared images with and without power line conductors. Those were annotated segmenting power lines from the background at a pixel level. Image sizes are 128
×
128 and 512
×
512. Within the same context, the VEPL Dataset Cano-Solis et al. (2023) is an SS dataset of vegetation encroachment in power line corridors, a relevant and unexplored topic of power line inspection. Its annotations are the vegetation mask locations in each image.

2.1.3Transmission Tower Dataset
The Tower dataset Bian et al. (2019) is an OD dataset consisting of 1,300 images of power line transmission towers collected from videos and pictures from the internet in different scenarios. Image size varies for each sample.

2.1.4Multi-category Datasets
\ac
ttpla Abdelfattah et al. (2020) is an \acis dataset with 1,100 aerial images of size 3840
×
2160 and two types of power line components: transmission towers and power line conductors. Each power line conductor and transmission tower in every image is annotated at a pixel level as individual instances, totalling 8,987. As for \acstnplad Vieira-e-Silva et al. (2021), it consists of 133 aerial images annotated for OD with five power line components: Transmission Tower, Insulator, Tower ID Plate, Yoke, and Stockbridge Damper. The image sizes vary between 5472
×
3078 or 5472
×
3648, which allows for a high density of instances per image, making a total of 2,409. It is worth noting that ImageNet Deng et al. (2009) has 1,290 images of transmission towers and power lines with annotated bounding boxes. Most images are not aerial and capture the front of the transmission towers. Finally, PLT-AI Furnas Dataset de Oliveira et al. (2022) contains 6,295 images with 17,808 annotations for OD. It has five different objects: Insulator, Baliser, Bird Nest, Separator and Stockbridge. Besides Bird Nest, they also have fault data sample annotations, which is quite valuable.

InsPLAD has 10,607 images, all in 1920
×
1080 resolution, with more than 28 thousand annotated power line components, categorised into 17 unique classes for the task of Object Detection. Besides that, power line components from five categories were cropped following a squared proportion (equal height and width) and annotated at an image level regarding their operational conditions: whether they are normal or present some defect. This last annotated data was divided into two ways for training and testing: one for a traditional supervised Image Classification (IC) task and the other for an unsupervised Anomaly Detection (AD) task. Finally, InsPLAD contains more images, annotations, components, defects, and vision tasks than all current public power line-related datasets.

Table 2:Public power line datasets. The defective assets column corresponds to the number of asset categories with defective samples available. Vision tasks: OD - Object Detection; SS - Semantic Segmentation; IS - Instance Segmentation; IC - Image Classification; AD - Anomaly Detection
Dataset	 
Power line
asset classes
Annotations	Images	 
Defective
asset classes
Vision tasks
Power line dataset (2017)
1	4,200	4,200	0	SS
Tomaszewski et al. (2018)
1	2,630	2,630	0	OD
Tower dataset (2019)
1	Undisclosed	1,300	0	OD
CPLID (2020)	1	1,569	848	1	OD
TTPLA (2020)	2	8,987	1,100	0	IS
STN PLAD (2021)	5	2,409	133	0	OD
PLT-AI (2022)	5	17,808	6,295	4	OD
InsPLAD (Ours)
17	28,933	10,607	5	 
OD, IC & AD
2.2Inspection methods
Here, we discuss suitable methods to address the challenges imposed by the datasets described above.

An Object Detection approach is a natural choice to address datasets such as CPLID, Tomaszewski et al., the Tower datasets, and STN PLAD, since their annotations are bounding boxes. A straightforward way is to use general-purpose object detection methods, such as \acssd Liu et al. (2016), \acyolo Redmon and Farhadi (2018), Faster \acrcnn Ren et al. (2015), and others. Such object detection datasets would fine-tune those object detectors, which were pre-trained with a large-scale multi-purpose dataset such as MS COCO Lin et al. (2014b). On the other hand, some works propose customised methods to address those datasets. For instance, in the CPLID work Tao et al. (2020), the authors also proposed a method to inspect the annotated insulators with a two-step object detector. It is based on Convolutional Neural Networks (CNNs) to extract features from the images and propose regions of interest, similar to the aforementioned general-purpose object detectors. In the Tower dataset paper, the authors also propose an adaptation of the Faster R-CNN method named Tower R-CNN with fewer convolutional layers to meet their prediction speed and accuracy requirements.

For Semantic Segmentation problems such as in the Powerline dataset, the annotations highlight the pixels in the images representing a particular class, i.e., masks. For instance, the masks in the Powerline dataset represent the precise locations of power line cables. A popular way to address this kind of dataset is to fine-tune general-purpose segmentation methods such as Mask R-CNN He et al. (2017) and U-Net Ronneberger et al. (2015). Other works also propose focused techniques for this kind of dataset. For example, in the Powerline dataset Lee et al. (2017), the authors also propose a method of inspecting the power lines by classifying sub-regions using a sliding window and a CNN. Specifically, a sub-region is filtered out if it is classified into an image without any power line. In that case, if a sub-region is classified into an image containing a power line, then its feature maps of intermediate convolutional layers are combined to visualise the location of the power line.

For Instance Segmentation, there are also general-purpose methods. For example, the authors of TTPLA use the Yolact Bolya et al. (2019) method for real-time instance segmentation with different backbones to create a baseline for that dataset. Other general-purpose instance segmentation methods, such as Mask R-CNN and CenterMask Lee and Park (2020), could also be applied.

Our proposed dataset, InsPLAD, has three different types of annotations. Bounding boxes for Object Detection and Image-level labels for Image Classification and Anomaly Detection. Object Detection methods were already presented. Regarding Image Classification, CNNs, e.g., ResNet He et al. (2016) and EfficientNet Tan and Le (2019), are still quite popular to approach this problem. Other recently popular methods for image classification are the ones based on Vision Transformers, such as the Swin-Transformer Liu et al. (2021). \acfcn, as \acmlp, still present results compatible with state-of-the-art, as seen in Tolstikhin et al. (2021).

Finally, Unsupervised Anomaly Detection is a recent research topic in Computer Vision. Methods to address it are usually based on Autoencoders Bergmann. et al. (2019), \acgan Zaheer et al. (2020), and Normalizing flows Rudolph et al. (2021); Rudolph et al. (2022), the last one being more recent and showing promising results.

3Dataset Description
This section describes the InsPLAD dataset and its divisions for object detection, fault classification, and anomaly detection. Initially, the process of capturing images in active power transmission lines using drones is described in Subsection 3.1. Next, the Subsection 3.2 discusses the annotation process of the images. Finally, in Subsection 3.3, the generated datasets are presented and discussed.

3.1Data Collection Protocol
Before starting the data collection, the main assets that may present problems were defined in cooperation with a power line company. The captured overline power line is a high-voltage 500 kV transmission line from the northeast region of Brazil. From the collaboration, 17 power line components were selected: Spiral Damper, Stockbridge Damper, Glass Insulator, Glass Insulator’s Big Shackle, Glass Insulator’s Small Shackle, Glass Insulator’s Tower Shackle, Lightning Rod’s Shackle, Lightning Rod’s Suspension, Tower ID Plate, Polymer Insulator, Polymer Insulator’s Lower Shackle, Polymer Insulator’s Upper Shackle, Polymer Insulator’s Tower Shackle, Spacer, Vari-grip, Yoke, Yoke’s Suspension.

After defining the assets that would receive attention during data collection, a capture protocol was developed to ensure the standardization of captured photos. The protocol aims to ensure that all highlighted components have images centring and standardise how the images should be captured to improve the inspection process. After that definition, the data collection protocol was implemented and refined between October and December 2020, using the DJI Matrice 210 V2 drone coupled to the Zenmuse z30 camera, which produces images with Full HD resolution (1920
×
1080). Two qualified drone pilots participated in a training session with researchers to follow the capture protocol. The pilots then captured the images under the supervision of the researchers to make last-minute adjustments to the protocol. Images of 226 power transmission line towers were captured during the process. Towers vary between structural types and each asset class contained in the tower. A total of 10,607 images were captured that focused on the selected assets.

The captured power line has three phases, A, B, and C, with B being the middle one. Phases A and C are typically mirrored as they are on opposite sides. This protocol was studied, improved, and refined daily during the collection phase until the latest version was reached. The detailed final protocol is presented below.

1. Photo of Tower ID Plate: One angle - One image (10 to 15 m away)
Refer to caption
Figure 1:ID Plate
2. PHASE A
(a) Stockbridge Damper set: One angle, framing on the tower’s side - One image (20 to 25 m)
Refer to caption
Figure 2:Stockbridge Damper
(b) Yoke Suspension: One angle, framing on the tower’s side - Two images, One per suspension pair (10 m)
Refer to caption
Figure 3:Yoke Suspension
(c) Yoke: One angle, framing on the tower’s side - One image (15 m)
Refer to caption
Figure 4:Yoke
(d) Polymer/Glass Insulator Lower/Small Shackle: One angle, framing on the tower’s side - One image (20 m)
Refer to caption
Figure 5:Polymer Insulator Lower Shackle
(e) Polymer/Glass Insulator:
i. An angle, wide frame on the tower’s side to show which Insulator will be detailed - One image (15 m)
ii. One angle, strict framing in the body of the Insulator - Four images, i.e. the Insulator is divided into at least Four images (25 m)
Refer to caption
Figure 6:Polymer Insulator
(f) Polymer/Glass Insulator Upper/Big Shackle: One angle, side framing - One image (10 m)
Refer to caption
Figure 7:Polymer Insulator Upper Shackle
3. Vari-grip on the left (if the tower has Vari-grips): One angle, diagonal framing - One image (9 m)
Refer to caption
Figure 8:Vari-grip
4. Lightning Rod Suspension/Shackle
Refer to caption
Figure 9:Lightning Rod Suspension
(a) Farther Lightning Rod Suspension/Shackle (opposite side): One angle, diagonal framing - One image (15 m)
(b) Closest Lightning Rod Suspension/Shackle: One angle, side framing - One image (15 m)
5. PHASE B: Repeat the same procedure as in PHASE A, only with the drone positioned in the front frame between the tower Lightning Rods. Distances are the same as in PHASE A: only change the sides but keep the angles on the opposite side.
6. Vari-grip on the right side: One angle, diagonal framing - One image (10 m)
7. PHASE C: Repeat the same procedure as in PHASE A, only now with the drone on the other side of the tower. Again, distances are the same as in PHASE A: The Vari-grip can be captured in PHASE A and PHASE C, always from the side to maintain safety, seeking a better angle with zoom and brightness adjustments. Repeat Step 5, only now positioning the drone on the other side of the tower. Again, distances are the same as in PHASE A.
3.2Annotation Process
The data was annotated by two annotators who received instructions from specialists about the names of the assets and regions of interest, which should be contained in the detection bounding boxes. Those bounding boxes are already presented in the data collection protocol above. In addition, all annotators were in constant contact with each other to ensure standardization in the data annotation process. The bounding box annotations were made with the help of the LabelImg tool Tzutalin (2015). This part of the dataset was named InsPLAD-det. All 17 asset classes and samples from InsPLAD-det can be seen in Figure 10.

Refer to caption
(a)
Refer to caption
(b)
Refer to caption
(c)
Refer to caption
(d)
Refer to caption
(e)
Refer to caption
(f)
Refer to caption
(g)
Refer to caption
(h)
Refer to caption
(i)
Refer to caption
(j)
Refer to caption
(k)
Refer to caption
(l)
Refer to caption
(m)
Refer to caption
(n)
Figure 10:All 17 assets of the InsPLAD-det. [R] and [G] indicate the colours of the bounding box of that asset in the image, [R] being Red and [G] Green. The assets are: (a) Damper - Spiral [R], (b) Damper - Stockbridge [R] and Yoke Suspension [G], (c) Glass Insulator [R], (d) Glass Insulator Big Shackle [R], (e) Glass Insulator Small Shackle [R], (f) Glass Insulator Tower Shackle [R], (g) Lightning Rod Shackle [R], (h) Lightning Rod Suspension [R], (i) Tower ID Plate [R], (j) Polymer Insulator [R] and Polymer Insulator Upper Shackle [G], (k) Polymer Insulator Lower Shackle [R] and Yoke [G], (l) Polymer Insulator Tower Shackle [R], (m) Spacer [R], (n) Vari-grip [R]. Best seen in colour
After creating InsPLAD-det, it was noticed that some asset classes had many faults, even for an active power line. An annotation process for the asset conditions with relevant defective data was carried out for this. These annotations enable the application of methods that seek to identify defects in the assets of power lines. With annotations regarding the state of the assets, it is possible to use image classification techniques to classify them according to their condition and point out defects. The same annotators from InsPLAD-det conducted the process. They also met constantly to ensure standardization in the annotation process.

The detected assets were cut from the original image using the regions of interest information contained in the InsPLAD-det’s bounding boxes to annotate the assets’ states. Cropped from the original images, the new images were grouped by asset class before starting the classification step regarding their status. The Pigeon tool Germanidis (2017) was used to help annotate the assets instances states. For each image analysed, the annotators classified its state as normal or the name of the defect found. With this process completed, the InsPLAD-fault dataset was created. InsPLAD-fault assets are shown in Figure 11. It shows the five asset classes that make up the dataset. In the first row, there are samples of assets under normal conditions, and in the second, defective samples.

Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Figure 11:All five asset classes of InsPLAD-fault. The first row contains images of the assets in their normal state, and the second contains the defective assets. From left to right, the assets in the images are Yoke Suspension (normal/rusty), Glass Insulator (normal/missing cap), Polymer Insulator Upper Shackle (normal/rusty), Lightning Rod Suspension (normal/rusty), and Vari-grip (normal/bird’s nest). Vari-grip can also be rusty
3.3Properties
After building the InsPLAD-det detection dataset, it was possible to obtain the dataset analytics. Table 3 shows essential dataset details that include the names of all assets, the total number of images containing each asset, the total number of annotation instances for each asset, and the average size of the annotations. It is possible to notice a total of 28,933 annotations in the dataset, a large number compared with other related datasets Lee et al. (2017); Tomaszewski et al. (2018); Tao et al. (2020); Bian et al. (2019); Abdelfattah et al. (2020); Vieira-e-Silva et al. (2021).

The images from each transmission tower were grouped during the data split to avoid similar images, e.g., the same object from different angles, in the InsPLAD-det training and test sets. Thus, 80% of the transmission towers were selected for training and 20% for testing. Also, the division was done carefully to evenly distribute each tower type and asset class between training and testing.

Table 3:Description of InsPLAD-det. The table shows the 17 assets in InsPLAD-det, the number of images containing each asset, the total number of annotations for each asset, the average size of annotations in squared pixels, and the division of training/test samples
Asset category
  	 
Number of
images
 	 
Number of
annotations
 	 
Average annotation
size (px²)
 	 
Train/test
samples
 
Damper - Spiral
  	943	1020	518.12 
±
 192.73	831/189
Damper - Stockbridge
  	1761	6953	162.12 
±
 58.25	5699/1254
Glass Insulator
  	2778	2978	546.08 
±
 206.05	2015/963
Glass Insulator Big Shackle
  	152	259	473.10 
±
 186.35	110/149
Glass Insulator Small Shackle
  	143	263	361.12 
±
 181.44	128/135
Glass Insulator Tower Shackle
  	106	195	204.29 
±
 82.68	98/97
Lightning Rod Shackle
  	112	195	286.54 
±
 110.28	170/25
Lightning Rod Suspension
  	709	710	363.26 
±
 166.37	618/92
Tower ID Plate
  	242	242	611.30 
±
 166.86	198/44
Polymer Insulator
  	3173	3244	534.65 
±
 233.90	2389/855
Pol. Insulator Lower Shackle
  	1760	1842	153.21 
±
 86.44	1460/382
Pol. Insulator Upper Shackle
  	1691	1692	392.55 
±
 160.92	1315/377
Pol. Insulator Tower Shackle
  	57	57	182.13 
±
 58.87	47/10
Spacer	93	94	463.99 
±
 182.75	72/22
Vari-grip	560	1008	461.75 
±
 139.54	846/162
Yoke	1661	1661	753.20 
±
 290.48	1343/318
Yoke Suspension	2716	6520	221.11 
±
 110.90	5270/1250
From the InsPLAD-det, it was possible to build the InsPLAD-fault with the classes that defects were observed. InsPLAD-fault has the properties shown in Table 4. Unlike InsPLAD-det, InsPLAD-fault has five component classes, since not all InsPLAD-det’s 17 classes had sufficient samples – the majority had no sample at all – of defective components. InsPLAD-fault’s five classes are divided into a set of normal and defective images. For Lightning Rod Suspension, Polymer Insulator Upper Shackle, Vari-grip, and Yoke classes, samples with rusty components were found. For the Glass Insulator class, there were missing cap faults. Additionally, for the Vari-grip, the presence of a bird’s nest was also found in some samples, so it is the only class with two types of faults noted. Figure 11 shows the examples of defects mentioned.

Due to the low number of faults in power lines, an unsupervised anomaly detection approach is also adopted to detect faults. Therefore, InsPLAD-fault was adapted into two approaches: supervised fault classification and unsupervised anomaly detection. The data was split to have sample representativeness in both training and testing sets and minimise the impact of unbalance between the normal and faulty classes. For this, only a smaller set of normal images was selected to compose the training set of assets under normal conditions. Table 4 shows the data split for both approaches. The main difference between both approaches is that, for anomaly detection, defective examples do not appear in the training set, only in the test set. Therefore, having more normal examples in anomaly detection training does not imply class imbalance. The other difference is that, for the fault classification approach, the faulty samples of the training sets were augmented by a factor of ten using traditional data augmentation techniques through the Albumentations library Buslaev et al. (2020). This is done to produce balanced training sets.

Table 4:Total images and distribution for training and testing for both InsPLAD-fault approaches. All assets contain only one fault/anomaly type except the Vari-grip, which had its faults divided into two categories separated by a forward slash (bird’s nest/rust)
Asset category	Fault classification	Anomaly detection
Train	Test	Train	Test
Normal	Fault	Normal	Fault	Normal	Normal	Anomaly
Glass Insulator	691	60	29	30	2298	581	90
Lightning Rod Suspension	348	30	231	20	462	117	50
Pol. Insulator Upper Shackle
742	48	31	33	935	235	102
Vari-grip	358	48/28	238	23/20	477	114	63/48
Yoke Suspension	299	29	5742	20	4834	1207	49
4Benchmark
We conducted several experiments with state-of-the-art methods for object detection, image classification, and anomaly detection to serve as a benchmark on InsPLAD. It is intended to provide a baseline for future work. The experiments were conducted in the Ubuntu 20.04 OS. For the ones in Subections 4.1 and 4.2.1 - Supervised fault classification, an Nvidia RTX 3080Ti GPU was used, and in 4.2.2 - Unsupervised anomaly detection, an Nvidia RTX 2080Ti was used.

4.1InsPLAD-det: Asset Detection
The first challenge imposed by our dataset is detecting power line assets in UAV-captured RGB images, which is an object detection task. We fine-tune popular and state-of-the-art object detectors, namely Faster R-CNN Ren et al. (2015), SSD Liu et al. (2016), YOLOv3 Redmon and Farhadi (2018), RetinaNet Lin et al. (2017), Cascade R-CNN Cai and Vasconcelos (2018), TOOD Feng et al. (2021) and DetectoRS Qiao et al. (2021).We also choose to experiment with older methods, such as YOLOv3, Faster R-CNN and SSD, because they are popular in power line inspection research Liu et al. (2020).

4.1.1Setup
All object detection methods were evaluated using their implementations from the MMDetection toolbox by OpenMMLab Chen et al. (2019), based on the PyTorch framework Paszke et al. (2019), and publicly available on GitHub MMDetection Contributors (2018). They were initialised from publicly available pre-trained weights provided by the MMDetection toolbox and fine-tuned until loss stabilization. The methods were trained in their standard configurations of parameters and hyperparameters. This is summarised in Table 5.

Table 5:Experimental configurations of the object detection methods
Method	Backbone	 
Input size
Epochs	 
Batch size
Learning Rate
SSD512	VGG16	
512
×
512
12	64	0.002
RetinaNet	 
ResNeXt-101
1333
×
800
12	16	0.01
YOLOv3	 
Darknet-53 608
608
×
608
12	64	0.001
TOOD	 
ResNet-101
1333
×
800
24	16	0.01
Faster R-CNN
ResNet-101
1333
×
800
12	16	0.02
Cascade R-CNN
ResNeXt-101
1333
×
800
12	16	0.02
DetectoRS	 
Cascade + ResNet-50
1333
×
800
12	16	0.02
Finally, the best-evaluated checkpoint was selected for further comparison by the end of each training.

The primary metric used for comparison is the Box AP from MS COCO, also known as AP (with \aciou as 
0.50
:
0.95
), currently the main object detection metric. AP50 and AP75 are used as secondary metrics. We also present weight size in megabytes and throughput, the latter being the average number of inferences the model makes per second during the test. This last information is relevant considering a real-time inspection scenario or a scenario with limited resources.

4.1.2Results and Discussion
Firstly, Table 6 shows the Box AP results for each object detector detailed per asset category. The variations between class AP values are due to object complexity, object size relative to the original image size from which it has been cropped, and the amount of training and testing samples. In addition, these variations could be responsible for the absence of an obvious lead detector for all classes, which can be further investigated.

Table 6:Box AP for each of the seventeen asset categories in every experimented object detection method. The best results for each class are highlighted in bold
Asset category
SSD
RetinaNet
YOLOv3
TOOD
Faster
R-CNN
Cascade
R-CNN
DetectoRS
Damper - Spiral
0.870	0.862	0.650	0.959	0.895	0.918	0.945
Damper - Stockbridge
0.815	0.845	0.701	0.857	0.835	0.842	0.848
Glass Insulator
0.803	0.838	0.752	0.889	0.840	0.880	0.893
Gl. Ins. Big Shackle
0.320	0.226	0.0012	0.184	0.137	0.203	0.248
Gl. Ins. Small Shackle
0.0016	0.275	0.189	0.214	0.264	0.280	0.270
Gl. Ins. Tower Shackle
0.377	0.373	0.246	0.392	0.360	0.433	0.413
Lightning Rod Shackle
0.566	0.547	0.402	0.569	0.490	0.561	0.595
Lightning Rod Susp.
0.869	0.928	0.774	0.928	0.911	0.914	0.911
Tower ID Plate
0.967	0.983	0.725	0.984	0.952	0.978	0.990
Polymer Insulator
0.850	0.953	0.785	0.951	0.921	0.950	0.954
Pol. Ins. Lower Shackle
0.578	0.639	0.453	0.637	0.611	0.644	0.648
Pol. Ins. Upper Shackle
0.796	0.855	0.715	0.872	0.835	0.846	0.857
Pol. Ins. Tower Shackle
0.507	0.531	0.377	0.371	0.500	0.405	0.528
Spacer	0.332	0.486	0.083	0.367	0.410	0.456	0.487
Vari-grip	0.914	0.948	0.826	0.954	0.930	0.941	0.954
Yoke	0.832	0.866	0.676	0.880	0.855	0.880	0.864
Yoke Suspension	0.808	0.853	0.769	0.856	0.842	0.855	0.855
Average	0.674	0.706	0.546	0.698	0.682	0.705	0.721
 
Table 7 summarises the performance of all object detectors. DetectoRS achieves the best performance considering Box AP and AP75 metrics, with a 0.721 Box AP and a 0.749 AP75; RetinaNet reaches second best in both of them, while it achieves the best result related to AP50 with 0.891; in this case, DetectoRS reaches second best as well as SSD and Cascade R-CNN. DetectoRS achieving the best overall performance is within expectations since it is a state-of-the-art multi-stage object detector. However, the performance of RetinaNet is the second-best overall, surpassing newer one- and multi-stage techniques.

On the other hand, SSD produces the fastest and most lightweight model, with 48.3 inferences per second during the test. Regarding performance, surprisingly, SSD reaches the same result as DectectoRS in terms of AP50, presenting a high potential to be used in real-time power line inspections due to its good performance and fast inference compared to the other methods.

Table 7:Performance, weight size in megabytes, and throughput results from state-of-the-art object detection methods. The first four methods correspond to one-stage object detection methods, while the last three are multi-stage approaches. Results in bold are the best for each aspect
Method	Reference	 
Weights size
(MB)
Inferences s-1	 
N° of
parameters
Box AP	AP50	AP75
SSD	ECCV16	215.2	48.3	36M	0.674	0.885	0.712
RetinaNet	ICCV17	756.2	11.7	56.3M	0.706	0.891	0.741
YOLOv3	arXiv18	493.3	41.6	61.9M	0.546	0.825	0.603
TOOD	ICCV21	427.7	13.2	53.3M	0.698	0.861	0.718
Faster R-CNN	NeurIPS15	483.3	18.2	60.5M	0.682	0.874	0.723
Cascade R-CNN	CVPR18	704.3	14.0	87.8M	0.705	0.885	0.733
DetectoRS	CVPR21	990.9	8.8	123.4M	0.721	0.885	0.749
Figure 12 shows the corresponding Precision-Recall curves for each asset and object detection method. The Dampers, Insulators, Tower ID Plate, Vari-grip, Lighting Rod Suspension, Yoke, and Yoke Suspension presented the best performance by the object detectors. At the same time, most types of Shackles and the Spacer had lower results. That can be explained mainly by the difference in the number of samples of each object. Almost all the objects the detectors perform best have the highest amount of samples in the dataset, which is expected since most Deep Learning methods are data-hungry. However, the Polymer Insulator Lower Shackle does not follow this pattern. It has a high amount of samples, but the object detectors do not present a good performance compared to other objects with many samples. That may be because different Polymer Insulator Lower Shackles instances have slight variations that do not justify separating them into sub-classes.

Refer to caption
Figure 12:Precision-Recall curves for each InsPLAD-det asset and all evaluated object detection methods
Figure 13 shows the detection results of DetectoRS, which is the best-performing object detector in terms of Box AP. The first sample is a case where the detector is successful, and the other is an example of a failure. It can be noticed in the first sample that it performs well in a sample with multiple objects of different classes, including partially occluded objects, e.g., the Stockbridge Damper near the centre of the image. On the other hand, the fail case shows four objects that were not detected, two of them completely unoccluded: the Polymer Insulator Lower Shackle below the left Polymer Insulator and the Yoke Suspension below the Polymer Insulator to the right. That might be due to lighting issues since the objects are darker than usual or some minor variations in their structure compared to other instances.

Refer to caption
(a)Successful case
Refer to caption
(b)Failure case
Figure 13:DetectoRS results for two test set samples. 13(a) is a successful case with multiple assets from five classes: Stockbridge Damper, Yoke Suspension, Yoke, Polymer Insulator, and Polymer Insulator Lower Shackle. In 13(b), two unoccluded objects were not detected: the Polymer Insulator Lower Shackle below the left Polymer Insulator and the Yoke Suspension below the right Polymer Insulator
4.2InsPLAD-fault
4.2.1Supervised Fault Classification
The second challenge presented by our dataset is an image classification task: given a detected power line asset, classify it regarding its apparent conditions. The asset conditions are varied, e.g., rust, missing cap, and being covered by a bird’s nest; their extension depends on the asset category. We found those conditions in five of the seventeen classes detected in InsPLAD. For this purpose, we train popular and state-of-the-art supervised image classification methods, namely EfficientNet Tan and Le (2019), ResNet He et al. (2016), ResNeXt Xie et al. (2017), MLP Mixer Tolstikhin et al. (2021) and Swin Transformer Liu et al. (2021). The first three are CNN-based methods, while the other two use different approaches. Swin Transformer is based on Vision Transformers, which have recently become popular, while MLP-Mixer is a new computer vision method, purely based on Multi-layer Perceptrons, achieving competitive results on image classification benchmarks Tolstikhin et al. (2021).

Setup
Here, traditional data augmentation was used, increasing the training data by ten times, as seen in Subsection 3.3. Finally, the following operations were randomly applied using the Albumentations library Buslaev et al. (2020): brightness and contrast alteration, histogram equalization, colour alteration by PCA, blurring, sharpening, horizontal mirroring, and rotation.

All object detection methods were evaluated using their implementations from the MMClassification toolbox, also by OpenMMLab, and also publicly available on GitHub MMClassification Contributors (2020). They were initialised from publicly available pre-trained weights provided by the MMClassification toolbox and fine-tuned until loss stabilization. The methods were trained in their standard configurations of parameters and hyperparameters for EfficientNet-B2, ResNeXt-101 32x8d, ResNet-101 8xb32, MLP Mixer-Base 64xb64, and Swin Transformer-Tiny 16xb64, as found in MMClassification. For these default MMClassification settings, the models were trained in 100 epochs, input images with a resolution of 224x224, a batch size of 32, and a Learning Rate of 0.1, except for Swin Transformer, which required 300 training epochs, batch size of 64 and Learning Rate of 0.001.

Finally, the best-evaluated checkpoints were selected for further comparison by the end of training. Balanced accuracy is used for comparison since some test sets are imbalanced Brodersen et al. (2010). Additionally, the size of weights and throughput were also registered.

Results and Discussion
Table 8 shows the results for each image classification method. EfficientNet achieves the best performance with a 0.954 average balanced accuracy. Swin Transformer presents a superior balanced accuracy in three out of five asset categories, with MLP-Mixer obtaining the highest for the Glass Insulator class and EfficientNet for the Vari-grip. Swin Transformer is the state-of-the-art method here, however, its average balanced accuracy is not able to surpass two other methods. On the other hand, MLP-Mixer is not based on CNNs or Vision Transformers and achieves performance comparable to the state-of-the-art.

Regarding inference speed, all models show high potential to be used in real time. However, EfficientNet produces the lightest model approximately five times compared with the second most lightweight. In a real-time inspection scenario with several power line assets, this lightweight property may be an essential feature to save resources. Another advantage of EfficientNet in this scenario is its performance. In our comparison, EfficientNet achieves a 0.954 overall balanced accuracy, being the best overall by a small margin.

Table 8:Fault classification balanced accuracy comparison between image classification methods for each asset class. Model size and throughput are also shown. The best results are highlighted in bold
ResNet
ResNeXt
EfficientNet
MLP-Mixer
Swin
Transformer
Reference	CVPR16	CVPR17	ICML19	NeurIPS21	ICCV21
Weights size (MB)	357	712	74	719	341
Inferences s-1 	587	347	544	589	587
N° of parameters	44.5M	88.8M	9.1M	59.9M	28.3M
Glass Insulator
0.833	0.811	0.866	0.916	0.883
Lightning Rod Susp.
0.989	0.926	0.996	0.987	0.998
Pol. Ins. Upper Shackle
0.832	0.876	0.985	0.955	0.985
Vari-grip	0.690	0.500	0.953	0.907	0.889
Yoke Suspension
0.861	0.832	0.970	0.992	0.999
Average	0.841	0.789	0.954	0.951	0.951
Figure 14 shows an example of a misclassified Glass Insulator by the EfficientNet method, which is the best-performing supervised fault classifier, on average. However, the method incorrectly classified it as a normal Glass Insulator, not a faulty one. That may be due to the Glass Insulator perspective, where the missing cap gaps are less evident than in a frontal view. However, the three missing caps are still recognizable by a specialist and even by the average human observer.

Refer to caption
Figure 14:Failure case of EfficientNet for a test sample of InsPLAD-fault. It was incorrectly classified as a defect-free Glass Insulator
4.2.2Unsupervised Anomaly Detection
The third and final challenge covered by InsPLAD is an anomaly detection task: detect on an image level whether a (cropped) power line asset is normal or anomalous. In this task, the model is only trained with normal objects and tested with normal and anomalous objects. Here, all the asset conditions are treated as anomalies. The same five asset categories in the subsection above are selected since anomalous samples are required for achieving quantitative results when testing. For this task, we selected and fine-tuned four anomaly detection methods for our analysis. We chose the Autorencoder 
L
2
 Bergmann. et al. (2019), because it was the best-performing method used in MVTec AD paper Bergmann et al. (2019) to benchmark their proposed anomaly detection dataset. Secondly, OGNet Zaheer et al. (2020) was selected to check if one-class classifiers could yield better results given the uncontrolled environment property provided by InsPLAD. Finally, DifferNet Rudolph et al. (2021) is a recent method that presents good performance on the MVTec AD dataset benchmark, and CS-Flow Rudolph et al. (2022) is a state-of-the-art anomaly detection method developed from DifferNet.

The first one is based on Convolutional Autoencoders Goodfellow et al. (2016) and reconstructs patches of 
128
×
128
 employing a per-pixel 
L
2
 distance Loss with a latent space dimension of 100. On the other hand, OGNet is based on GANs. The discriminator task is to distinguish between good and bad quality reconstructions yielded by a new and an old state of the generator, respectively. That allows the discriminator to learn the small nuances of an anomaly, outputting what is considered an anomaly score. Finally, the CS-Flow and DifferNet methods are based on CNNs and normalizing flows. They use the CNN feature extraction to obtain an anomaly score through normalizing flows. CS-Flow and DifferNet are specific defect detection methods tested on a state-of-the-art anomaly detection dataset Bergmann et al. (2019).

Setup
Both feature-extraction CNNs in CS-Flow and DifferNet, EfficientNet-B5, and AlexNet, respectively, are initialised using weights pre-trained with ImageNet. The codes used in this experiment were obtained in their official repositories by their authors. Table 9 shows the experimental configurations for each method. As in the previous tasks, the best-evaluated checkpoints during training were selected for the baselines.

Table 9:Experimental configurations of the anomaly detection methods
Method	Input size	Epochs	Batch size	Learning rate
Autoencoder	128x128	200	16	0.0002
OGNet	45x45	200	16	0.001
DifferNet	448x448	192	24	0.0002
CS-Flow	768x768	240	14	0.0002
The metric adopted for the baselines is the \acauroc curve, which is the primary metric used by the community. The ROC curve is plotted with TPR against the FPR by varying the binary classification threshold for anomalous and normal using the anomaly score. As the curve is constructed from the TPR and FPR rates, it is robust to unbalanced datasets of anomalous and normal samples, which commonly occurs in anomaly detection application scenarios. Therefore, when there is a threshold value capable of separating true positives from true negatives, the curve nears the upper left corner, resulting in the area under the curve close to 
1.0
, which means a good result. The curve nears a diagonal straight line in the opposite case, indicating a random classifier. In addition, the size of weights and throughput were also registered.

Results and Discussion
Table 10 summarises the results obtained for the anomaly detection methods in InsPLAD-anomaly. DifferNet achieves the best average AUROC result with 0.905, but CS-Flow obtains the best results in three out of five asset categories and comes in a close second with a 0.903 average AUROC. Due to that, there is no clear advantage to a single method. However, the approach using CNN feature extraction with normalizing flows adopted by both presents a superiority over the others. For the OGNet network, the average obtained among the five objects was 0.635. Based on adversarial training, the network obtained a better result than Autoencoder 
L
2
, even with fewer parameters and used images with just 45 pixels on each side. Resizing, characteristic of the network pipeline, reduces the image, so the network needs fewer parameters to process it but loses information in the image. The loss of information makes training challenging to learn some object features, as occurred for the Lighting Rod Suspension, where the network training could not converge and classified every test sample as anomalous. Although Autoencoder 
L
2
 is the lightest model, there is a more than 30 percentage points gap between it and the best-performing methods. For each object and anomaly detection method, the corresponding ROC curves are given in Figure 15.

In terms of inference throughput, all methods achieved satisfactory results. Although DifferNet is the largest model, it is the best-performing one, making 70 inferences per second. It is a promising alternative in a real-time inspection scenario, considering other tasks must be executed in the inspection pipeline.

Table 10:AUROC comparison between anomaly detection methods for each evaluated asset. The size of the models in megabytes and throughput are also presented. The best results are in bold
Autoencoder 
L
2
OGNet
DifferNet
CS-Flow
Reference	VISAPP19	CVPR20	WACV21	WACV22
Weights size (MB)	4	52	933	602
∼
727
Inferences s-1 	96	80	70	39
N° of parameters	28.8M	12.9M	233.1M	148.2M
Glass Insulator	0.614	0.618	0.824	0.841
Lightning Rod Suspension
  	0.512	0.500	0.991	0.966
Pol. Insulator Upper Shackle
  	0.555	0.612	0.901	0.884
Vari-grip	0.628	0.775	0.906	0.915
Yoke Suspension
  	0.551	0.672	0.905	0.907
Average	0.572	0.635	0.905	0.903
Refer to caption
Figure 15:ROC curves, which plot True Positive Rate against False Positive Rate for each InsPLAD-fault category and all evaluated anomaly detection methods
4.3InsPLAD Strengths and Limitations
The InsPLAD is proposed in a context where multiple works acknowledge a lack of public benchmarks for power line inspection. As presented in the Introduction, each challenge proposed by InsPLAD aims to fill the gaps mentioned in Table 1. Firstly, InsPLAD-det presents many categories of power line assets under various external conditions, orientations, and subtypes, with more than 28k samples. That is much higher than the current public power line asset data. However, the amount of samples for each class varies significantly, producing an unbalanced dataset. To help with the imbalance between classes, data augmentation techniques, as also done in MVTec AD, help reduce the impact caused by the frequency diversity of objects in transmission line towers. An example of performing data augmentation in this work is applying rotation and mirroring randomly to generate new images of specific classes.

InsPLAD-fault and InsPLAD-anomaly share similar advantages and drawbacks. Public data regarding faulty power line assets is even rarer than datasets for object detection or instance segmentation of power line assets. Both InsPLAD datasets present different faults/anomalies for different assets, which is unprecedented. They contain six defects: four are corrosion, one is a broken component, and one is the presence of a bird’s nest. However, these defects are spread over five assets; four have just one type of fault, and one has two types: corrosion and bird’s nest presence. That arises from the inherent aspect that finding anomalies/defects in real-world operating power lines is challenging, as they receive predictive maintenance and the problems are scarce and quickly corrected. Therefore, it would be ideal to have more than one type of defect per category, as seen in other anomaly detection datasets, for instance, MVTec AD, which focuses on objects in controlled environments and has on average five per category Bergmann et al. (2019). As an anomaly is defined as something that deviates from the standard, containing any deviation type, it is important to have variations with diversity to ensure that the proposed models learned the object’s standard and not how to identify a specific problem. With defects similar to each other or just one type of defect for the asset, it allows a model to learn to recognize that defect and not the normal pattern expected from the asset. As shown by Campos et al. (2016) for Outlier detection, a similar problem, datasets with diversity in their outlier examples allow a new method to demonstrate its superiority over existing methods by consistently detecting the outliers in these datasets.

Finally, both InsPLAD-det and InsPLAD-fault share similar challenges with other computer vision application areas, such as agriculture with the detection of plant diseases Singh et al. (2020), in transport with the detection of pedestrians and vehicles Caesar et al. (2020), in sport with the analysis of player performance Wu et al. (2022), among others. All these areas and the Powerline inspection have challenges in common, such as small and big objects, multiple objects per image, uncontrolled background, environment and point-of-views, and objects with multiple instances.

5Conclusions
This work proposes InsPLAD, a dataset for power line asset inspection made from UAV images from real-world inspections. InsPLAD is organised to meet different steps on a standard power line inspection: object detection for the detection of power line components in UAV images, image classification for identifying defects in the detected components, and unsupervised anomaly detection, which has the same objective as the latter but treats the defects as anomalies and no defective data is used for training. InsPLAD is the largest public power line-related dataset regarding power line components, number of images, annotations, defects, and vision tasks. A benchmark with state-of-the-art and popular methods for object detection, image classification, and anomaly detection is also provided to serve as a baseline. It shows room for improvement in all three steps. We hope InsPLAD sparks future research on visual inspection in the power line area and other areas with similar challenges.

InsPLAD opens lots of possibilities for future research. More specifically, new methods for unsupervised anomaly detection in the wild seem quite attractive since defective component data is scarce, and new datasets alone will not be able to cover all possible defect types due to the uncontrolled environment nature of the problem. Besides that, InsPLAD shows room for improvement in it. Another problem is the large size of the best-performing anomaly detection methods, which heavily hinders their deployment into production, even more so considering that a different model is used for every component type. Finally, new datasets for power line inspection are also welcome, especially ones with data on multiple defective components.

Acknowledgement(s)
The authors acknowledge the financial support of STN - Sistema de Transmissão Nordeste S.A. through the ANEEL R&D Program for the development of the research project entitled: “PD-04825-0006/2019: Inspeção com Drones por Meio do Acoplamento Eletrostático para Carregamento de Baterias em Voo e Uso de Aprendizagem de Máquina para Classificação Automática de Defeitos”.

Disclosure of interest
The authors report no conflict of interest.

Funding
This work was supported by STN - Sistema de Transmissão Nordeste S.A. through the ANEEL R&D Program for the development of the research project under grant code PD-04825-0006/2019; Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) under Finance Code 001; and Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq).

Data availability statement
The data that support the findings of this study is openly available in Mendeley Data at https://data.mendeley.com/preview/5n3fjgvfyz?a=f68efe15-61d3-4a74-bc8a-d009e3cd3f95.

References
Abdelfattah et al. (2020)Abdelfattah, R., X. Wang, and S. Wang (2020, November).TTPLA: An aerial-image dataset for detection and segmentation of transmission towers and power lines.In Proceedings of the Asian Conference on Computer Vision (ACCV).
Bergmann et al. (2019)Bergmann, P., M. Fauser, D. Sattlegger, and C. Steger (2019, June).MVTec AD – a comprehensive real-world dataset for unsupervised anomaly detection.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Bergmann. et al. (2019)Bergmann., P., S. Löwe., M. Fauser., D. Sattlegger., and C. Steger. (2019).Improving unsupervised defect segmentation by applying structural similarity to autoencoders.In Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,, pp.  372–380. INSTICC: SciTePress.
Bian et al. (2019)Bian, J., X. Hui, X. Zhao, and M. Tan (2019).A monocular vision–based perception approach for unmanned aerial vehicle close proximity transmission tower inspection.International Journal of Advanced Robotic Systems 16(1), 1729881418820227.
Bolya et al. (2019)Bolya, D., C. Zhou, F. Xiao, and Y. J. Lee (2019, October).Yolact: Real-time instance segmentation.In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
Brodersen et al. (2010)Brodersen, K. H., C. S. Ong, K. E. Stephan, and J. M. Buhmann (2010).The balanced accuracy and its posterior distribution.In 2010 20th International Conference on Pattern Recognition, pp.  3121–3124.
Bruch et al. (2011)Bruch, M., V. Münch, M. Aichinger, M. Kuhn, M. Weymann, and G. Schmid (2011).Power blackout risks.In Cro Forum, pp.  28.
Buslaev et al. (2020)Buslaev, A., V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin (2020).Albumentations: Fast and flexible image augmentations.Information 11(2), n/a.
Caesar et al. (2020)Caesar, H., V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom (2020).nuscenes: A multimodal dataset for autonomous driving.In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.  11621–11631.
Cai and Vasconcelos (2018)Cai, Z. and N. Vasconcelos (2018, June).Cascade R-CNN: Delving into high quality object detection.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Campos et al. (2016)Campos, G. O., A. Zimek, J. Sander, R. J. Campello, B. Micenková, E. Schubert, I. Assent, and M. E. Houle (2016).On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study.Data mining and knowledge discovery 30, 891–927.
Cano-Solis et al. (2023)Cano-Solis, M., J. R. Ballesteros, and J. W. Branch-Bedoya (2023).Vepl dataset: A vegetation encroachment in power line corridors dataset for semantic segmentation of drone aerial orthomosaics.Data 8(8).
Chen et al. (2019)Chen, K., J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, and D. Lin (2019).MMDetection: Open mmlab detection toolbox and benchmark.
de Oliveira et al. (2022)de Oliveira, F. S., M. de Carvalho, P. H. T. Campos, A. Da Silva Soares, A. C. Júnior, and A. C. R. Da Silva Quirino (2022).Ptl-ai furnas dataset: A public dataset for fault detection in power transmission lines using aerial images.In 2022 35th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Volume 1, pp.  7–12.
Deng et al. (2009)Deng, J., W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei (2009).ImageNet: A large-scale hierarchical image database.In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp.  248–255.
Everingham et al. (2010)Everingham, M., L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman (2010).The pascal visual object classes (voc) challenge.International journal of computer vision 88(2), 303–338.
Feng et al. (2021)Feng, C., Y. Zhong, Y. Gao, M. R. Scott, and W. Huang (2021, October).TOOD: Task-aligned one-stage object detection.In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp.  3510–3519.
Germanidis (2017)Germanidis, A. (2017).Pigeon - Quickly annotate data on Jupyter.GitHub Repository 1, n/a.
Goodfellow et al. (2016)Goodfellow, I., Y. Bengio, and A. Courville (2016).Deep learning.MIT press.
He et al. (2017)He, K., G. Gkioxari, P. Dollar, and R. Girshick (2017, Oct).Mask r-cnn.In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
He et al. (2016)He, K., X. Zhang, S. Ren, and J. Sun (2016, June).Deep residual learning for image recognition.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Hu and Liu (2017)Hu, Y. and K. Liu (2017).Inspection and Monitoring Technologies of Transmission Lines with Remote Sensing.Academic Press.
Kuznetsova et al. (2020)Kuznetsova, A., H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, T. Duerig, and V. Ferrari (2020).The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.IJCV 88(7), 1956–1981.
Lee et al. (2017)Lee, S. J., J. P. Yun, H. Choi, W. Kwon, G. Koo, and S. W. Kim (2017).Weakly supervised learning with convolutional neural networks for power line localization.In 2017 IEEE Symposium Series on Computational Intelligence (SSCI), pp.  1–8.
Lee and Park (2020)Lee, Y. and J. Park (2020, June).Centermask: Real-time anchor-free instance segmentation.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Lei and Sui (2019)Lei, X. and Z. Sui (2019).Intelligent fault detection of high voltage line based on the faster r-cnn.Measurement 138, 379–385.
Li et al. (2020)Li, L., H. Wu, Y. Song, and Y. Liu (2020).A state-failure-network method to identify critical components in power systems.Electric Power Systems Research 181, 106192.
Lin et al. (2017)Lin, T.-Y., P. Goyal, R. Girshick, K. He, and P. Dollar (2017, Oct).Focal loss for dense object detection.In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
Lin et al. (2014a)Lin, T.-Y., M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick (2014a).Microsoft COCO: Common Objects in Context.In D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars (Eds.), Computer Vision – ECCV 2014, Cham, pp.  740–755. Springer International Publishing.
Lin et al. (2014b)Lin, T.-Y., M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick (2014b).Microsoft coco: Common objects in context.In D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars (Eds.), Computer Vision – ECCV 2014, Cham, pp.  740–755. Springer International Publishing.
Liu et al. (2023)Liu, J., G. Xie, J. Wang, S. Li, C. Wang, F. Zheng, and Y. Jin (2023).Deep industrial image anomaly detection: A survey.arXiv preprint arXiv:2301.11514 2.
Liu et al. (2016)Liu, W., D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg (2016).SSD: Single shot multibox detector.In B. Leibe, J. Matas, N. Sebe, and M. Welling (Eds.), Computer Vision – ECCV 2016, Cham, pp.  21–37. Springer International Publishing.
Liu et al. (2020)Liu, X., X. Miao, H. Jiang, and J. Chen (2020).Data analysis in visual power line inspection: An in-depth review of deep learning for component detection and fault diagnosis.Annual Reviews in Control 50, 253–277.
Liu et al. (2021)Liu, Z., Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo (2021, October).Swin Transformer: Hierarchical vision transformer using shifted windows.In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp.  10012–10022.
MMClassification Contributors (2020)MMClassification Contributors (2020, 7).OpenMMLab’s Image Classification Toolbox and Benchmark.
MMDetection Contributors (2018)MMDetection Contributors (2018, 8).OpenMMLab Detection Toolbox and Benchmark.
Nguyen et al. (2018)Nguyen, V. N., R. Jenssen, and D. Roverso (2018).Automatic autonomous vision-based power line inspection: A review of current status and the potential role of deep learning.International Journal of Electrical Power & Energy Systems 99, 107–120.
Paszke et al. (2019)Paszke, A., S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019).PyTorch: An imperative style, high-performance deep learning library.In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Advances in Neural Information Processing Systems, Volume 32. Curran Associates, Inc.
Qiao et al. (2021)Qiao, S., L.-C. Chen, and A. Yuille (2021, June).DetectoRS: Detecting objects with recursive feature pyramid and switchable atrous convolution.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.  10213–10224.
Rahmani et al. (2013)Rahmani, A., M. Khadem, E. Madreseh, H.-A. Aghaei, M. Raei, and M. Karchani (2013).Descriptive study of occupational accidents and their causes among electricity distribution company workers at an eight-year period in iran.Safety and health at work 4(3), 160–165.
Redmon and Farhadi (2018)Redmon, J. and A. Farhadi (2018).YOLOv3: An incremental improvement.
Ren et al. (2015)Ren, S., K. He, R. Girshick, and J. Sun (2015).Faster R-CNN: Towards real-time object detection with region proposal networks.In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (Eds.), Advances in Neural Information Processing Systems, Volume 28. Curran Associates, Inc.
Ronneberger et al. (2015)Ronneberger, O., P. Fischer, and T. Brox (2015).U-net: Convolutional networks for biomedical image segmentation.In N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Cham, pp.  234–241. Springer International Publishing.
Rudolph et al. (2021)Rudolph, M., B. Wandt, and B. Rosenhahn (2021, January).Same same but DifferNet: Semi-supervised defect detection with normalizing flows.In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp.  1907–1916.
Rudolph et al. (2022)Rudolph, M., T. Wehrbein, B. Rosenhahn, and B. Wandt (2022, January).Fully convolutional cross-scale-flows for image-based defect detection.In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp.  1088–1097.
Ruszczak et al. (2023)Ruszczak, B., P. Michalski, and M. Tomaszewski (2023).Overview of image datasets for deep learning applications in diagnostics of power infrastructure.Sensors 23(16).
Siddiqui et al. (2018)Siddiqui, Z. A., U. Park, S.-W. Lee, N.-J. Jung, M. Choi, C. Lim, and J.-H. Seo (2018).Robust powerline equipment inspection system based on a convolutional neural network.Sensors 18(11), 3837.
Singh et al. (2020)Singh, D., N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra (2020).Plantdoc: A dataset for visual plant disease detection.In Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, CoDS COMAD 2020, New York, NY, USA, pp.  249–253. Association for Computing Machinery.
Tan and Le (2019)Tan, M. and Q. Le (2019, 09–15 Jun).EfficientNet: Rethinking model scaling for convolutional neural networks.In K. Chaudhuri and R. Salakhutdinov (Eds.), Proceedings of the 36th International Conference on Machine Learning, Volume 97 of Proceedings of Machine Learning Research, pp.  6105–6114. PMLR.
Tao et al. (2020)Tao, X., D. Zhang, Z. Wang, X. Liu, H. Zhang, and D. Xu (2020).Detection of power line insulator defects using aerial images analyzed with convolutional neural networks.IEEE Transactions on Systems, Man, and Cybernetics: Systems 50(4), 1486–1498.
Tolstikhin et al. (2021)Tolstikhin, I. O., N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, M. Lucic, and A. Dosovitskiy (2021).Mlp-mixer: An all-mlp architecture for vision.In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan (Eds.), Advances in Neural Information Processing Systems, Volume 34, pp.  24261–24272. Curran Associates, Inc.
Tomaszewski et al. (2018)Tomaszewski, M., B. Ruszczak, and P. Michalski (2018).The collection of images of an insulator taken outdoors in varying lighting conditions with additional laser spots.Data in Brief 18, 765–768.
Tzutalin (2015)Tzutalin, D. (2015).LabelImg.GitHub Repository 6, n/a.
Vieira-e-Silva et al. (2021)Vieira-e-Silva, A. L. B., H. de Castro Felix, T. de Menezes Chaves, F. P. M. Simões, V. Teichrieb, M. M. dos Santos, H. da Cunha Santiago, V. A. C. Sgotti, and H. B. D. T. L. Neto (2021).STN PLAD: A dataset for multi-size power line assets detection in high-resolution uav images.In 2021 34th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pp.  215–222.
Wu et al. (2022)Wu, D., H. Zhao, X. Bao, and R. P. Wildes (2022).Sports video analysis on large-scale data.In European Conference on Computer Vision, pp.  19–36. Springer.
Xia et al. (2022)Xia, X., X. Pan, N. Li, X. He, L. Ma, X. Zhang, and N. Ding (2022).GAN-based anomaly detection: A review.Neurocomputing 493, 497–535.
Xie et al. (2017)Xie, S., R. Girshick, P. Dollar, Z. Tu, and K. He (2017, July).Aggregated residual transformations for deep neural networks.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Yang et al. (2019)Yang, Y., L. Wang, Y. Wang, and X. Mei (2019).Insulator self-shattering detection: a deep convolutional neural network approach.Multimedia Tools and Applications 78(8), 10097–10112.
Zaheer et al. (2020)Zaheer, M. Z., J.-H. Lee, M. Astrid, and S.-I. Lee (2020, June).Old Is Gold: Redefining the adversarially learned one-class classifier training paradigm.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Zhang et al. (2019)Zhang, H., M. Sun, Y. Ji, S. Xu, and W. Cao (2019).Learning-based object detection in high resolution uav images: An empirical study.In 2019 IEEE 17th International Conference on Industrial Informatics (INDIN), Volume 1, pp.  886–889.
Zhang et al. (2023)Zhang, Z., Z. Zhao, X. Zhang, C. Sun, and X. Chen (2023).Industrial anomaly detection with domain shift: A real-world dataset and masked multi-scale reconstruction.
◄ ar5iv homepage Feeling
lucky? Conversion
report Report
an issue View original
on arXiv►
Copyright Privacy Policy Generated on Tue Feb 27 20:32:16 2024 by LaTeXMLMascot Sammy