/******************************************************************************
 * Video Analytics - Project
 *
 * Date: 5th May, 2014
 *
 * Author: Sandeep Prakash
 *
 * Description: 
 * 1.	Work on real-time video (i.e., not captured video)
 * 2.	Video should have few persons (say 2 to 4) moving around and one or 
 *      two of them should be wearing UTD logo T-shirts.
 * 3.	UTD T-shirts do have different types of UTD logos. You can choose to 
 *      track any one (or more) of these logos. The choice of logos is yours. 
 *      But during the demo you should have at least 1 person wearing that logo
 *      T-shirt (and that can be you yourself).
 * 4.	Of the few persons seen on the video, track only the person wearing UTD 
 *      shirt, i.e., put a bounding box on the entire person wearing the T-shirt
 *      (not just the T-shirt alone).
 * 5.	Bounding box on the entire person wearing T-shirt is a requirement.
 * 6.	At the end (which can be decided by just pressing a key), display a 
 *      graphic representation of how the UTD T-shirt person was moving, i.e., 
 *      the path of the person’s movement. This can be done by displaying a 
 *      rectangular box (for the captured video space) and drawing lines that 
 *      would connect the points of the center of the bounding box in different 
 *      video frames.
 * 7.	Only one such graphic representation of the UTD T-shirt person’s movement
 *      path is needed – this path will show the cumulative movement over the 
 *      entire video clip captured.
 *
 * Usage:
 *  1. The following code was testing using OpenCV version opencv-2.4.8.
 *  2. Visual Studio 2012 on 64-bit machine was used.
 *  3. Option to learn from available images on disk or from camera capture.
 *
 * References:
 * 1. http://docs.opencv.org/doc/tutorials/tutorials.html
 * 2. http://docs.opencv.org/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html
 * 3. https://www.youtube.com/watch?v=ZXn69V-1kEM
 * 4. https://www.youtube.com/watch?v=fXKw0rt-NEs - The people tracking system is implemented with GMM Background subtraction and Kalman filtering
 * 5. https://www.youtube.com/watch?v=__0qu3mpDSA - OpenCV Tutorial: Detecting People | packtpub.com
 * 6. https://www.youtube.com/watch?v=mFnZ2bqMnSI - More People Detection with OpenCV
 * 7. http://www.magicandlove.com/blog/2011/08/26/people-detection-in-opencv-again/ - People Detection in OpenCV again
 * 8. http://stackoverflow.com/questions/11696393/opencv-2-4-cascadeclassified-detectmultiscale-arguments
 * 9. http://stackoverflow.com/questions/19480172/opencv-human-body-tracking
 * 10. https://github.com/Itseez/opencv/tree/master/data/haarcascades
 * 11. http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html - Cascade Classifier
 * 12. https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_mcs_upperbody.xml - 22x20 Head and shoulders detector
 *
 *****************************************************************************/

/********************************** INCLUDES **********************************/
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/********************************* CONSTANTS **********************************/

/*********************************** MACROS ***********************************/
#define MAX_FILEPATH_LEN						(1024)
#define CONFIG_FILENAME							".//config//config.txt"
#define MAX_IMAGES_TO_TRAIN						(50)

/******************************** ENUMERATIONS ********************************/

/************************* STRUCTURE/UNION DATA TYPES *************************/
typedef struct _CONFIG_X
{
	int iMAX_IMAGES_TO_TRAIN;
	int iVIDEO_CAP_DEVICE;
	double dVIDEO_CAP_FPS;
	char cVIDEO_CAP_WINDOW_NAME[MAX_FILEPATH_LEN];
	int iVIDEO_CAP_WIDTH_PIXELS;
	int iVIDEO_CAP_HEIGHT_PIXELS;
	char cTRAINING_INDEX_FILE_PATH[MAX_FILEPATH_LEN];
	char cTRAINING_INDEX_FILE_NAME[MAX_FILEPATH_LEN];
	char cTRAINING_UPPER_BODY_CASCADE_FILENAME[MAX_FILEPATH_LEN];
} CONFIG_X;

VideoCapture cameraCapture;

vector < KeyPoint > trainingImageKeyPointObject[MAX_IMAGES_TO_TRAIN];

Mat trainedImageObject[MAX_IMAGES_TO_TRAIN];

SurfFeatureDetector surfFeatureDetector;

SurfDescriptorExtractor surfDescriptorExtractor;

Mat trainingImageGrayScale[MAX_IMAGES_TO_TRAIN];

vector < Point2f > obj_corners (4);

FlannBasedMatcher flannBasedMatcher;

Mat capturedImageRgbFrame;

Mat capturedImageGrayFrame;

vector < KeyPoint > capturedImageKeyPoints;

Mat capturedImageDescriptors;

vector<vector<DMatch > > capturedImageMatchesVector[MAX_IMAGES_TO_TRAIN];

vector < DMatch > trainedImage2CapturedImageMatches[MAX_IMAGES_TO_TRAIN];

Mat trainedImage2CapturedImageMatchedImage[MAX_IMAGES_TO_TRAIN];

CascadeClassifier upperBodyCascade;

std::vector<Rect> bodies;

int maxBodiesRectangleAreaIndex = 0;

list<Point2f> trackingHistory;

/************************ STATIC FUNCTION PROTOTYPES **************************/

/****************************** LOCAL FUNCTIONS *******************************/
const string objectDetectionAndTrackingWindowName = "Person Detection and Tracking";
int thresholdMin = 0;
int thresholdMax = 6;

int boxThresholdMin = 0;
int boxThresholdMax = 100;

void loadConfig (CONFIG_X *px_config)
{
	FILE *fp = NULL;
	char *key;
	char *val;
	char line[MAX_FILEPATH_LEN*2] = {0};

	fp = fopen(CONFIG_FILENAME, "r");
	if(!fp)
	{
		printf ("Error: Could not find config file: %d\n", CONFIG_FILENAME);
		goto CLEAN;
	}
	while(!feof(fp))
	{
		fscanf(fp, "%s", line);

		key = strtok(line, "=");
		val = strtok(NULL, "=");

		if (0 == strcmp (key, "MAX_IMAGES_TO_TRAIN"))
		{
			px_config->iMAX_IMAGES_TO_TRAIN = atoi (val);
		}
		else if (0 == strcmp (key, "VIDEO_CAP_DEVICE"))
		{
			px_config->iVIDEO_CAP_DEVICE = atoi (val);
		}
		else if (0 == strcmp (key, "VIDEO_CAP_FPS"))
		{
			px_config->dVIDEO_CAP_FPS = atof (val);
		}
		else if (0 == strcmp (key, "VIDEO_CAP_WINDOW_NAME"))
		{
			strncpy (px_config->cVIDEO_CAP_WINDOW_NAME, val, 
				sizeof(px_config->cVIDEO_CAP_WINDOW_NAME));
		}
		else if (0 == strcmp (key, "VIDEO_CAP_WIDTH_PIXELS"))
		{
			px_config->iVIDEO_CAP_WIDTH_PIXELS = atoi (val);
		}
		else if (0 == strcmp (key, "VIDEO_CAP_HEIGHT_PIXELS"))
		{
			px_config->iVIDEO_CAP_HEIGHT_PIXELS = atoi (val);
		}
		else if (0 == strcmp (key, "TRAINING_INDEX_FILE_PATH"))
		{
			strncpy (px_config->cTRAINING_INDEX_FILE_PATH, val, 
				sizeof(px_config->cTRAINING_INDEX_FILE_PATH));
		}
		else if (0 == strcmp (key, "TRAINING_INDEX_FILE_NAME"))
		{
			strncpy (px_config->cTRAINING_INDEX_FILE_NAME, val, 
				sizeof(px_config->cTRAINING_INDEX_FILE_NAME));
		}
		else if (0 == strcmp (key, "TRAINING_UPPER_BODY_CASCADE_FILENAME"))
		{
			strncpy (px_config->cTRAINING_UPPER_BODY_CASCADE_FILENAME, val, 
				sizeof(px_config->cTRAINING_UPPER_BODY_CASCADE_FILENAME));
		}
	}
	fclose(fp);

CLEAN:
	return;
}

void createTrackbars()
{
    namedWindow(objectDetectionAndTrackingWindowName, WINDOW_AUTOSIZE);
	char TrackbarName[50];
	sprintf( TrackbarName, "Threshold: ", thresholdMin);
  
    createTrackbar( "Threshold: ", objectDetectionAndTrackingWindowName, 
		&thresholdMin, thresholdMax, NULL);

	createTrackbar("Box Thres: ", objectDetectionAndTrackingWindowName, 
		&boxThresholdMin, boxThresholdMax, NULL );
}

int cameraCaptureForTraining (CONFIG_X *px_config, unsigned int ui_idx)
{
	Mat bgrFrameCamera;
	bool bStatus = false;

	/**************************************************************************
	 * Start Video Capture
	 *************************************************************************/
	/*
	 * Open the camera device.
	 */
    VideoCapture cameraCapture(px_config->iVIDEO_CAP_DEVICE);
	if(false == cameraCapture.isOpened())
	{
		printf ("Camera Device Open Failed!");
        return -1;
	}

	/*
	 * Set the desired FPS.
	 */
	bStatus = cameraCapture.set(CV_CAP_PROP_FPS, px_config->dVIDEO_CAP_FPS);
	if (false == bStatus)
	{
		return -1;
	}

	/*
	 * Capture a new frame from the camera. Use this frame to get the frame  
	 * size which is used in the future while opening a video file for writing.
	 */
	cameraCapture >> bgrFrameCamera;

    while (true)
    {
		/*
		 * Capture one frame from camera.
		 */
        cameraCapture >> bgrFrameCamera;

		imshow (px_config->cVIDEO_CAP_WINDOW_NAME, bgrFrameCamera);

        if(waitKey(30) >= 0) break;
    }

	/*
	 * After the user presses a key, that frame will be used as the trained 
	 * image.
	 */
	cvtColor (bgrFrameCamera, trainingImageGrayScale[ui_idx], CV_RGB2GRAY);

	// Release resources.
	cameraCapture.release ();
	/**************************************************************************
	 * End Video Capture
	 *************************************************************************/
	return 0;
}

unsigned int getNoOfFilesInIndexFileForTraining (CONFIG_X *px_config, char* indexFilePath, char *indexFileName)
{
	FILE *fp = NULL;
	char filename[1024];
	char indexFilename[1024];
	unsigned int ui_count = 0;

	sprintf_s (indexFilename, sizeof(indexFilename), "%s/%s", indexFilePath, indexFileName);

	fp = fopen(indexFilename, "r");
	if(!fp)
	{
		printf ("Error: Could not find training index file: %d\n", indexFilename);
		return 0;
	}
	while(!feof(fp))
	{
		fscanf(fp, "%s", filename);
		ui_count++;
	}
	fclose(fp);

	if (ui_count > MAX_IMAGES_TO_TRAIN)
	{
		printf ("Max trainable images: %d. Defaulting to that! "
			"First %d images in index will be trained\n", 
			MAX_IMAGES_TO_TRAIN, MAX_IMAGES_TO_TRAIN);
		ui_count = MAX_IMAGES_TO_TRAIN;
	}
	return ui_count;
}

bool readImageForTraining(CONFIG_X *px_config, char *pc_folder_path, FILE *fp, unsigned int ui_idx) 
{
	bool b_success = false;
	char filename[1024];
	char filepath[1024];

	fscanf(fp, "%s", filename);
	sprintf_s (filepath, sizeof(filepath), "%s/%s", pc_folder_path, filename);
	printf ("Processing file: %s\n", filepath);
	Mat bgrFrameImage = imread(filepath);
	if (!bgrFrameImage.empty()) 
	{
		cvtColor (bgrFrameImage, trainingImageGrayScale[ui_idx], 
			CV_RGB2GRAY);
		imshow (px_config->cVIDEO_CAP_WINDOW_NAME, bgrFrameImage);
		waitKey(1);
		b_success = true;
	}
	else
	{
		printf ("Warning: Could not read image: %s\n", filepath);
		b_success = false;
	}
	return b_success;
}

int trainAndLearnDesiredObject (CONFIG_X *px_config, 
	unsigned int *pui_no_training_images,
	int i_option)
{
	int i_ret_val = -1;
	int minHess = 0;
	unsigned int ui_i = 0;
	unsigned int ui_no_training_images = 0;
	FILE *fp = NULL;
	bool b_error = false;

	switch (i_option)
	{
	case 1:
		{
			ui_no_training_images = *pui_no_training_images;
			break;
		}
	case 2:
		{
			ui_no_training_images = getNoOfFilesInIndexFileForTraining (px_config,
				px_config->cTRAINING_INDEX_FILE_PATH, px_config->cTRAINING_INDEX_FILE_NAME);

			char indexFilename[1024];
			sprintf_s (indexFilename, sizeof(indexFilename), "%s/%s",
				px_config->cTRAINING_INDEX_FILE_PATH, px_config->cTRAINING_INDEX_FILE_NAME);
			fp = fopen(indexFilename, "r");
			if(!fp)
			{
				printf ("Error: Could not find training index file: %d\n", 
					indexFilename);
				i_ret_val = -1;
				goto CLEAN_RETURN;
			}
			break;
		}
	default:
		{
			printf ("Invalid Option! Try Again!");
			i_ret_val = -1;
			goto CLEAN_RETURN;
		}
	}

	namedWindow(px_config->cVIDEO_CAP_WINDOW_NAME, 1);

	/**************************************************************************
	 * Train and learn the object of interest.
	 *************************************************************************/
	for (ui_i = 0; ui_i < ui_no_training_images; ui_i++)
	{
		switch (i_option)
		{
		case 1:
			{
				cameraCaptureForTraining (px_config, ui_i);
				break;
			}
		case 2:
			{
				if (!readImageForTraining(px_config, px_config->cTRAINING_INDEX_FILE_PATH, fp, ui_i))
				{
					b_error = true;
				}
				break;
			}
		default:
			{
				printf ("Invalid Option! Try Again!");
				i_ret_val = -1;
				goto CLEAN_RETURN;
			}
		}

		if (true == b_error)
		{
			break;
		}		

		minHess = 2000;
		surfFeatureDetector = SurfFeatureDetector(minHess);
		surfFeatureDetector.detect (
		   trainingImageGrayScale[ui_i], 
		   trainingImageKeyPointObject[ui_i]);
		surfDescriptorExtractor.compute (
		   trainingImageGrayScale[ui_i], 
		   trainingImageKeyPointObject[ui_i], 
		   trainedImageObject[ui_i]);

		//_sleep (500);
	}

	destroyWindow(px_config->cVIDEO_CAP_WINDOW_NAME);

	switch (i_option)
	{
	case 1:
		{
			*pui_no_training_images = ui_i;
			break;
		}
	case 2:
		{
			fclose(fp);
			*pui_no_training_images = ui_i;
			break;
		}
	default:
		{
			printf ("Invalid Option! Try Again!");
			i_ret_val = -1;
			goto CLEAN_RETURN;
		}
	}

	char cascadeFilename[1024];
	sprintf_s (cascadeFilename, sizeof(cascadeFilename), "%s/%s",
		px_config->cTRAINING_INDEX_FILE_PATH, px_config->cTRAINING_UPPER_BODY_CASCADE_FILENAME);
	b_error = upperBodyCascade.load(cascadeFilename);
	if(false == b_error)
	{
		printf("Error loading haar cascade file: %s\n", cascadeFilename); 
		i_ret_val = -1; 
	}
	else
	{
		i_ret_val = 0;
	}
CLEAN_RETURN:
	return i_ret_val;
}

int initiateCameraCapture (CONFIG_X *px_config)
{
	int i_ret_val = -1;
	
	bool bStatus = false;

	/**************************************************************************
	 * Start Video Capture
	 *************************************************************************/
	/*
	 * Open the camera device.
	 */
    cameraCapture = VideoCapture(px_config->iVIDEO_CAP_DEVICE);
	if(false == cameraCapture.isOpened())
	{
		printf ("Camera Device Open Failed!");
		i_ret_val = -1;
        goto CLEAN_RETURN;
	}

	/*
	 * Set the desired FPS.
	 */
	bStatus = cameraCapture.set(CV_CAP_PROP_FPS, px_config->dVIDEO_CAP_FPS);
	if (false == bStatus)
	{
		i_ret_val = -1;
		goto CLEAN_RETURN;
	}

	bStatus = cameraCapture.set (CV_CAP_PROP_FRAME_WIDTH, 
		px_config->iVIDEO_CAP_WIDTH_PIXELS);
	if (false == bStatus)
	{
		i_ret_val = -1;
		goto CLEAN_RETURN;
	}

    bStatus = cameraCapture.set (CV_CAP_PROP_FRAME_HEIGHT, 
		px_config->iVIDEO_CAP_HEIGHT_PIXELS);
	if (false == bStatus)
	{
		i_ret_val = -1;
	}
	else
	{
		i_ret_val = 0;
	}
CLEAN_RETURN:
	return i_ret_val;
}

int initiatePlottingObject (CONFIG_X *px_config)
{
	int i_ret_val = -1;

	obj_corners [0] = cvPoint (0, 0);
	obj_corners [1] = cvPoint (trainingImageGrayScale[0].cols, 0);
	obj_corners [2] = cvPoint (trainingImageGrayScale[0].cols, 
		trainingImageGrayScale[0].rows);
	obj_corners [3] = cvPoint (0, trainingImageGrayScale[0].rows);
	i_ret_val = 0;
	return i_ret_val;
}

bool captureFrameAndDetect (CONFIG_X *px_config)
{
	bool b_found = false;

	cameraCapture >> capturedImageRgbFrame;
    cvtColor (capturedImageRgbFrame, capturedImageGrayFrame, CV_RGB2GRAY);

	surfFeatureDetector.detect (capturedImageGrayFrame, capturedImageKeyPoints);
    surfDescriptorExtractor.compute (capturedImageGrayFrame, 
		capturedImageKeyPoints, capturedImageDescriptors);

	if (NULL != capturedImageDescriptors.data)
	{
		b_found = true;
	}
	else
	{
		b_found = false;
	}
	return b_found;
}

void myPutText (Mat& img, const string& text, Point org)
{
	putText (img, text, org, CV_FONT_NORMAL, 2, 
				cvScalar (0, 0, 250), 1, CV_AA);
}

void matchDetectedFrame (CONFIG_X *px_config, 
						 unsigned int ui_no_training_images)
{
	double thresholdMatchingNN = 0.7;
	unsigned int ui_i = 0;
	char coordinates[256] = {0};

	/*
	 * Try to match with the trained images.
	 */
	for (ui_i = 0; ui_i < ui_no_training_images; ui_i++)
	{
		capturedImageMatchesVector[ui_i].clear();

		flannBasedMatcher.knnMatch (trainedImageObject[ui_i], 
			capturedImageDescriptors, capturedImageMatchesVector[ui_i], 2);	

		trainedImage2CapturedImageMatches[ui_i].clear();

		int maxVectorSize = min (capturedImageDescriptors.rows - 1, 
			(int) capturedImageMatchesVector[ui_i].size ());
		for (int i = 0; i < maxVectorSize; i++)
		{
			if ((capturedImageMatchesVector[ui_i] [i] [0].distance
				< thresholdMatchingNN * (capturedImageMatchesVector[ui_i] [i] [1].distance))
				&& ((int) capturedImageMatchesVector[ui_i] [i].size () <= 2
					&& (int) capturedImageMatchesVector[ui_i] [i].size () > 0))
			{
				trainedImage2CapturedImageMatches[ui_i].push_back (
					capturedImageMatchesVector[ui_i] [i] [0]);
			}
		}
		
		equalizeHist(capturedImageGrayFrame, capturedImageGrayFrame);
		upperBodyCascade.detectMultiScale (capturedImageGrayFrame, 
			bodies, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(45,80) );

		maxBodiesRectangleAreaIndex = -1;
		if (0 != bodies.size())
		{
			int maxRectangleArea = 0;
			for( size_t i = 0; i < bodies.size(); i++ )
			{
				Point center( bodies[i].x + bodies[i].width/2, bodies[i].y + bodies[i].height/2);

				sprintf_s(coordinates, sizeof(coordinates), "(%d, %d)", center.x, center.y);

				int curRectangleArea = bodies[i].width * bodies[i].height;

				if (curRectangleArea > maxRectangleArea)
				{
					maxRectangleArea = curRectangleArea;
					maxBodiesRectangleAreaIndex = i;
				}
			}

			Point center( bodies[maxBodiesRectangleAreaIndex].x + bodies[maxBodiesRectangleAreaIndex].width/2, 
				bodies[maxBodiesRectangleAreaIndex].y + bodies[maxBodiesRectangleAreaIndex].height/2);
			sprintf_s(coordinates, sizeof(coordinates), "(%d, %d)", center.x, center.y);
		}
		trainedImage2CapturedImageMatchedImage[ui_i] = capturedImageRgbFrame;
	}
}

/*
 * http://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
 */
int findLineIntersection (Point2f &line1_p1, Point2f &line1_p2, 
						  Point2f &line1_p3, Point2f &line1_p4, Point2f &intersection_p)
{
	int iRetVal = -1;
	int denominator = -1;

	denominator = ((line1_p1.x - line1_p2.x) * (line1_p3.y - line1_p4.y)) - 
		((line1_p1.y - line1_p2.y) * (line1_p3.x - line1_p4.x));

	if (0 == denominator)
	{
		iRetVal = -1;
		goto CLEAN_RETURN;
	}

	int intersection_p_x_num = 0;
	int intersection_p_y_num = 0;

	intersection_p_x_num = (((line1_p1.x * line1_p2.y) - (line1_p1.y * line1_p2.x)) * (line1_p3.x - line1_p4.x)) - 
		((line1_p1.x - line1_p2.x) * ((line1_p3.x * line1_p4.y) - (line1_p3.y * line1_p4.x)));

	intersection_p_y_num = (((line1_p1.x * line1_p2.y) - (line1_p1.y * line1_p2.x)) * (line1_p3.y - line1_p4.y)) - 
		((line1_p1.y - line1_p2.y) * ((line1_p3.x * line1_p4.y) - (line1_p3.y * line1_p4.x)));

	double x = (double) intersection_p_x_num / (double) denominator;
	double y = (double) intersection_p_y_num / (double) denominator;

	intersection_p.x = (float) x;
	intersection_p.y = (float) y;
	iRetVal = 0;
CLEAN_RETURN:
	return iRetVal;
}

double findLineSlope (Point2f &line1_p1, Point2f &line1_p2)
{
	if (line1_p2.x - line1_p1.x == 0)
	{
		return FLT_MAX;
	}
	else
	{
		return ((line1_p2.y - line1_p1.y) / (line1_p2.x - line1_p1.x));
	}
}

float getDistanceBetweenPoints (Point2f &point1,
								Point2f &point2)
{
	float x2MinusX1 = 0.0;
	float y2MinusY1 = 0.0;

	x2MinusX1 = point2.x - point1.x;
	y2MinusY1 = point2.y - point1.y;

	return sqrtf ((x2MinusX1 * x2MinusX1) + (y2MinusY1 * y2MinusY1));
}


// #define CONSIDER_SLOPE

int getShortestDistanceToBodyIndex (Point2f &detectedObjectCenter)
{
	int shortestDistanceToBodyIndex = -1;
	float minDistance = FLT_MAX;
	double maxSlope = 0.0;

	if (0 != bodies.size())
	{
		for(size_t i = 0; i < bodies.size(); i++)
		{
			Point2f bodyCenter((float) bodies[i].x + bodies[i].width/2, 
				(float) bodies[i].y + bodies[i].height/2);

			float distance = getDistanceBetweenPoints (bodyCenter,
								detectedObjectCenter);
#ifdef CONSIDER_SLOPE
			double slope = findLineSlope (bodyCenter, detectedObjectCenter);
#endif

			if (distance < minDistance)
			{
#ifdef CONSIDER_SLOPE
				if (slope > maxSlope)
				{
					maxSlope = slope;
					minDistance = distance;
					shortestDistanceToBodyIndex = i;
				}
#else
				minDistance = distance;
				shortestDistanceToBodyIndex = i;
#endif
				
			}
		}
	}

	return shortestDistanceToBodyIndex;
}

void showDetectedGoodFrame (CONFIG_X *px_config, unsigned int ui_no_training_images)
{
	Mat H;       
    vector < Point2f > obj;
    vector < Point2f > scene;
    vector < Point2f > scene_corners (4);
	int threshold = thresholdMin + 4;
	unsigned int ui_matches_max = 0;
	unsigned int ui_matches_max_idx = 0;
	unsigned int ui_i = 0;

	/*
	 * Get the image with almost the exact match. That will have the highest 
	 * number of matches. Use that image to render and draw matching lines 
	 * a box around the object.
	 */
	for (ui_i = 0; ui_i < ui_no_training_images; ui_i++)
	{
		if (trainedImage2CapturedImageMatches[ui_i].size () >= threshold)
		{
			if (trainedImage2CapturedImageMatches[ui_i].size () >= ui_matches_max)
			{
				ui_matches_max = (unsigned int) trainedImage2CapturedImageMatches[ui_i].size ();
				ui_matches_max_idx = ui_i;
			}
		}
	}

	if (trainedImage2CapturedImageMatches[ui_matches_max_idx].size () >= threshold)
	{
		for (unsigned int i = 0; i < trainedImage2CapturedImageMatches[ui_matches_max_idx].size (); i++)
		{
			int queryIdx = trainedImage2CapturedImageMatches[ui_matches_max_idx] [i].queryIdx;
			int trainIdx = trainedImage2CapturedImageMatches[ui_matches_max_idx] [i].trainIdx;
			obj.push_back (trainingImageKeyPointObject[ui_matches_max_idx] [queryIdx].pt);
			scene.push_back (capturedImageKeyPoints [trainIdx].pt);
		}

		H = findHomography (obj, scene, CV_RANSAC);

		perspectiveTransform (obj_corners, scene_corners, H);

		char coordinates[256] = {0};
		Point2f intersection_p;

		int iRetVal = findLineIntersection (scene_corners [0], scene_corners [2], 
						  scene_corners [1], scene_corners [3], intersection_p);

		int shortestDistanceToBodyIndex = getShortestDistanceToBodyIndex (intersection_p);

		if ((shortestDistanceToBodyIndex >= 0) && (0 == iRetVal))
		{
			Rect drawRectangle = Rect (bodies[shortestDistanceToBodyIndex]);

			drawRectangle.height = drawRectangle.height * 3;

			rectangle(capturedImageRgbFrame, drawRectangle, Scalar(0, 255, 0), 4);

			line (trainedImage2CapturedImageMatchedImage[ui_matches_max_idx], 
				scene_corners [0], scene_corners [2], Scalar (0, 255, 0), 1);
			line (trainedImage2CapturedImageMatchedImage[ui_matches_max_idx], 
				scene_corners [1], scene_corners [3], Scalar (0, 255, 0), 1);

			Point2f intersectionRectangle;

			Point2f line1P1 = Point (bodies[shortestDistanceToBodyIndex].x, bodies[shortestDistanceToBodyIndex].y);
			Point2f line2P1 = Point (bodies[shortestDistanceToBodyIndex].x + bodies[shortestDistanceToBodyIndex].width, 
				bodies[shortestDistanceToBodyIndex].y);
			Point2f line1P2 = Point (bodies[shortestDistanceToBodyIndex].x + bodies[shortestDistanceToBodyIndex].width, 
				bodies[shortestDistanceToBodyIndex].y + bodies[shortestDistanceToBodyIndex].height);
			Point2f line2P2 = Point (bodies[shortestDistanceToBodyIndex].x, 
				bodies[shortestDistanceToBodyIndex].y + bodies[shortestDistanceToBodyIndex].height);
			findLineIntersection (line1P1, line1P2, line2P1, line2P2, intersectionRectangle);

			if (trackingHistory.size() <= 100)
			{
				trackingHistory.push_back(intersectionRectangle);
			}
			else
			{
				trackingHistory.pop_front();
				trackingHistory.push_back(intersectionRectangle);
			}
		}
	}

	if (trackingHistory.size() > 0)
	{
		int i = 0;
		std::list<Point2f>::const_iterator iterator;
		for (iterator = trackingHistory.begin(); iterator != trackingHistory.end(); )
		{
			circle(trainedImage2CapturedImageMatchedImage[ui_matches_max_idx],
				*iterator, 10, Scalar (0, 255, 0));

			Point2f start = Point2f (*iterator);

			++iterator;

			if (iterator != trackingHistory.end())
			{
				Point2f end = Point2f (*iterator);
				line (trainedImage2CapturedImageMatchedImage[ui_matches_max_idx], 
					start, end, Scalar (0, 255, 0), 1);
			}
		}
	}
		
	imshow (objectDetectionAndTrackingWindowName, 
		trainedImage2CapturedImageMatchedImage[ui_matches_max_idx]);
}

void mainVideoCaptureLoop (CONFIG_X *px_config, unsigned int ui_no_training_images)
{
	bool b_found = false;

	while (1)
    {
        b_found = captureFrameAndDetect (px_config);

		if (true == b_found)
		{
			matchDetectedFrame (px_config, ui_no_training_images);

			showDetectedGoodFrame (px_config, ui_no_training_images);
		}
		if(waitKey(30) >= 0) 
		{
			break;
		}
    }
}

int main(int i_argc, char**ppc_argv)
{
	int i_ret_val = -1;
	unsigned int ui_no_training_images = 0;
	int i_option = -1;
	CONFIG_X x_config = {0};
	
	loadConfig (&x_config);

AGAIN:
	printf ("Training Option:\n"
		"1. Camera Capture\n"
		"2. Index File\n"
		"Enter Option: ");
	//scanf ("%d", &i_option);

	i_option = 2;
	switch (i_option)
	{
	case 1:
		{
			printf ("\n Enter number of images to be trained: ");
			scanf ("%d", &ui_no_training_images);

			if (ui_no_training_images > MAX_IMAGES_TO_TRAIN)
			{
				printf ("Max trainable images: %d. Defaulting to that!\n", MAX_IMAGES_TO_TRAIN);
				ui_no_training_images = MAX_IMAGES_TO_TRAIN;
			}
			break;
		}
	case 2:
		{
			break;
		}
	default:
		{
			printf ("Invalid Option! Try Again!");
			goto AGAIN;
		}
	}
			
	i_ret_val = trainAndLearnDesiredObject (&x_config, &ui_no_training_images, i_option);
	if (0 != i_ret_val)
	{
		printf ("trainAndLearnDesiredObject Failed: %d\n", i_ret_val);
        goto CLEAN_RETURN;
	}

	i_ret_val = initiateCameraCapture (&x_config);
	if (0 != i_ret_val)
	{
		printf ("initiateCameraCapture Failed: %d\n", i_ret_val);
        goto CLEAN_RETURN;
	}

	i_ret_val = initiatePlottingObject (&x_config);
    if (0 != i_ret_val)
	{
		printf ("iniatePlottingObject Failed: %d\n", i_ret_val);
        goto CLEAN_RETURN;
	}

	createTrackbars();

	mainVideoCaptureLoop (&x_config, ui_no_training_images);

    cameraCapture.release ();
	destroyWindow(objectDetectionAndTrackingWindowName);
	i_ret_val = 0;
CLEAN_RETURN:
	return i_ret_val;
}
