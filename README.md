# ch-cv-human-tracking
Tracking humans in live video.

Description
===========

1. Work on real-time video (i.e., not captured video)                               
2. Video should have few persons (say 2 to 4) moving around and one or              
     two of them should be wearing UTD logo T-shirts.                               
3. UTD T-shirts do have different types of UTD logos. You can choose to             
     track any one (or more) of these logos. The choice of logos is yours.       
     But during the demo you should have at least 1 person wearing that logo     
     T-shirt (and that can be you yourself).                                        
4. Of the few persons seen on the video, track only the person wearing UTD       
     shirt, i.e., put a bounding box on the entire person wearing the T-shirt    
     (not just the T-shirt alone).                                                  
5. Bounding box on the entire person wearing T-shirt is a requirement.              
6. At the end (which can be decided by just pressing a key), display a              
     graphic representation of how the UTD T-shirt person was moving, i.e.,      
     the path of the person's movement. This can be done by displaying a      
     rectangular box (for the captured video space) and drawing lines that       
     would connect the points of the center of the bounding box in different     
     video frames.                                                               
7. Only one such graphic representation of the UTD T-shirt person's movement  
     path is needed & this path will show the cumulative movement over the    
     entire video clip captured.                                                    
                                                                                 
Usage
=====

 1. The following code was testing using OpenCV version opencv-2.4.8.               
 2. Visual Studio 2012 on 64-bit machine was used.                                  
 3. Option to learn from available images on disk or from camera capture.           
                                                                                 
References
==========

1. http://docs.opencv.org/doc/tutorials/tutorials.html                              
2. http://docs.opencv.org/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html
3. https://www.youtube.com/watch?v=ZXn69V-1kEM                                      
4. https://www.youtube.com/watch?v=fXKw0rt-NEs - The people tracking system is implemented with GMM Background subtraction and Kalman filtering
5. https://www.youtube.com/watch?v=__0qu3mpDSA - OpenCV Tutorial: Detecting People | packtpub.com
6. https://www.youtube.com/watch?v=mFnZ2bqMnSI - More People Detection with OpenCV
7. http://www.magicandlove.com/blog/2011/08/26/people-detection-in-opencv-again/ - People Detection in OpenCV again
8. http://stackoverflow.com/questions/11696393/opencv-2-4-cascadeclassified-detectmultiscale-arguments
9. http://stackoverflow.com/questions/19480172/opencv-human-body-tracking           
10. https://github.com/Itseez/opencv/tree/master/data/haarcascades                  
11. http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html - Cascade Classifier
12. https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_mcs_upperbody.xml - 22x20 Head and shoulders detector
