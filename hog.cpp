#include "precomp.hpp"
#include <iterator>
#ifdef HAVE_IPP
#include "ipp.h"
#endif

/*
Size winSize;//���ڴ�С
Size blockSize;//Block��С
Size blockStride;//blockÿ���ƶ���Ȱ���ˮƽ�ʹ�ֱ��������
Size cellSize;//Cell��Ԫ��С
int nbins;//ֱ��ͼbin��Ŀ
int derivAperture;//��֪��ʲô��		?????????????????
double winSigma;//��˹�����ķ���
int histogramNormType;//ֱ��ͼ��һ�����ͣ����������
double L2HysThreshold;//L2Hys�����������ֵΪ0.2 
bool gammaCorrection;//�Ƿ�GammaУ�� 
vector<float> svmDetector;//�������
*/
namespace cv
{

size_t HOGDescriptor::getDescriptorSize() const
{
	//������ݵĺ�����
    //����2������Ǳ�֤block����������cell;��֤block�ڴ��������ƶ�������
    CV_Assert(blockSize.width % cellSize.width == 0 &&
        blockSize.height % cellSize.height == 0);
    CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
        (winSize.height - blockSize.height) % blockStride.height == 0 );
    //���ص�nbins��ÿ�������м�⵽��hog������ά��
    return (size_t)nbins*
        (blockSize.width/cellSize.width)*
        (blockSize.height/cellSize.height)*
        ((winSize.width - blockSize.width)/blockStride.width + 1)*
        ((winSize.height - blockSize.height)/blockStride.height + 1);
}

//winSigma������ʲô�����أ�
double HOGDescriptor::getWinSigma() const
{
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}

//svmDetector��HOGDescriptor�ڵ�һ����Ա��������������Ϊ����vector��
//��������hog��������svm����ʱ��ϵ����.
//�ú�������Ϊ���ʵ�ʺ�����ʲô�أ���֤��hog����������ͬ���������1����Ϊʲô
//���1Ҳ�����أ�
bool HOGDescriptor::checkDetectorSize() const
{
    size_t detectorSize = svmDetector.size(), descriptorSize = getDescriptorSize();
    //�����������һ��Ϊtrue����ʽΪtrue��ʵ���������һ��
	return detectorSize == 0 ||
        detectorSize == descriptorSize ||
        detectorSize == descriptorSize + 1;
}

void HOGDescriptor::setSVMDetector(InputArray _svmDetector)
{  
    //�����convertTo����ֻ�ǽ�ͼ��Mat���Ը��ģ�����˵ͨ������������ȵȡ�
    //�����ǽ������svmϵ������ȫ��ת���ɸ����͡�
    _svmDetector.getMat().convertTo(svmDetector, CV_32F);
    CV_Assert( checkDetectorSize() );
}

#define CV_TYPE_NAME_HOG_DESCRIPTOR "opencv-object-detector-hog"

//FileNode��opencv��core�е�һ���ļ��洢�ڵ��࣬����ڵ������洢��ȡ����ÿһ���ļ�Ԫ�ء�
//һ���Ƕ�ȡXML��YAML��ʽ���ļ�
//����Ϊ�ú����ǰ��ļ��ڵ��е����ݶ�ȡ������ĳ�Ա�����У����Ժ������治���йؼ���const
bool HOGDescriptor::read(FileNode& obj)
{
    //isMap()�������ж�����ڵ��ǲ���һ��ӳ�����ͣ������ӳ�����ͣ���ÿ���ڵ㶼��
    //һ�����ֶ�Ӧ��������������if�������þ������ȡ���ļ�node��һ��ӳ������
    if( !obj.isMap() )
        return false;
    //�������е�"winSize"��ָ������ΪwinSize��һ���ڵ㣬��Ϊ�Ѿ�֪����Щ�ڵ���mapping����
    //Ҳ����˵����һ����Ӧ�����֡�
    FileNodeIterator it = obj["winSize"].begin();
    //������>>Ϊ�ӽڵ��ж������ݣ������ǽ�itָ��Ľڵ��������ζ���winSize.width,winSize.height
    //����ļ�����书������
    it >> winSize.width >> winSize.height;
    it = obj["blockSize"].begin();
    it >> blockSize.width >> blockSize.height;
    it = obj["blockStride"].begin();
    it >> blockStride.width >> blockStride.height;
    it = obj["cellSize"].begin();
    it >> cellSize.width >> cellSize.height;
    obj["nbins"] >> nbins;
    obj["derivAperture"] >> derivAperture;
    obj["winSigma"] >> winSigma;
    obj["histogramNormType"] >> histogramNormType;
    obj["L2HysThreshold"] >> L2HysThreshold;
    obj["gammaCorrection"] >> gammaCorrection;
    obj["nlevels"] >> nlevels;
    
    //isSeq()���жϸýڵ������ǲ���һ������
    FileNode vecNode = obj["SVMDetector"];
    if( vecNode.isSeq() )
    {
        vecNode >> svmDetector;
        CV_Assert(checkDetectorSize());
    }
    //����Ķ���ȡ���˺�ͷ��ض�ȡ�ɹ���־
    return true;
}
    
void HOGDescriptor::write(FileStorage& fs, const String& objName) const
{
    //��objName�������뵽�ļ�fs��
    if( !objName.empty() )
        fs << objName;

    fs << "{" CV_TYPE_NAME_HOG_DESCRIPTOR
    //���漸�����ν�hog�������ڵı������뵽�ļ�fs�У���ÿ������ǰ������
    //һ�����������Ӧ�������Щ�ڵ���mapping���͡�
    << "winSize" << winSize
    << "blockSize" << blockSize
    << "blockStride" << blockStride
    << "cellSize" << cellSize
    << "nbins" << nbins
    << "derivAperture" << derivAperture
    << "winSigma" << getWinSigma()
    << "histogramNormType" << histogramNormType
    << "L2HysThreshold" << L2HysThreshold
    << "gammaCorrection" << gammaCorrection
    << "nlevels" << nlevels;
    if( !svmDetector.empty() )
        //svmDetector����ֱ���������У�Ҳ�ж�Ӧ�����֡�
        fs << "SVMDetector" << "[:" << svmDetector << "]";
    fs << "}";
}

//�Ӹ������ļ��ж�ȡ����
bool HOGDescriptor::load(const String& filename, const String& objname)
{
    FileStorage fs(filename, FileStorage::READ);
    //һ���ļ��ڵ��кܶ�Ҷ�ӣ�����һ���ļ��ڵ�����˺ܶ����ݣ����ﵱȻ�ǰ�����
    //HOGDescriptor��Ҫ�ĸ��ֲ����ˡ�
    FileNode obj = !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

//�����еĲ������ļ��ڵ����ʽд���ļ��С�
void HOGDescriptor::save(const String& filename, const String& objName) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    write(fs, !objName.empty() ? objName : FileStorage::getDefaultObjectName(filename));
}

//����HOG�����ӵ�c��
void HOGDescriptor::copyTo(HOGDescriptor& c) const
{
    c.winSize = winSize;
    c.blockSize = blockSize;
    c.blockStride = blockStride;
    c.cellSize = cellSize;
    c.nbins = nbins;
    c.derivAperture = derivAperture;
    c.winSigma = winSigma;
    c.histogramNormType = histogramNormType;
    c.L2HysThreshold = L2HysThreshold;
    c.gammaCorrection = gammaCorrection;
    //vector����Ҳ�����õȺŸ�ֵ
    c.svmDetector = svmDetector;
	c.nlevels = nlevels; 
} 

//img:ԭʼͼ��
//grad:��¼ÿ����������bin��Ӧ��Ȩ�صľ���,Ϊ��ֵ����Ȩֵ
//���Ȩֵ�ǹؼ���Ҳ�ܸ��ӣ�������˹Ȩ�أ����β�ֵ��Ȩ�أ��ڱ���������ֵ���Ƿ�ֵ������bin��Ĳ�ֵȨ��
//qangle:��¼ÿ�����ؽǶ�������bin��ŵľ���,��Ϊ2ͨ��,Ϊ�����Բ�ֵ
//paddingTL:Top��Left����������
//paddingBR:����ͬ��
//���ܣ�����img�����ź��ͼ����ÿ�����ص��ݶȺͽǶ�

//����ͼ��img���ݶȷ���ͼ��grad���ݶȷ���ͼ��qangle.
//paddingTLΪ��Ҫ��ԭͼ��img���Ͻ������ĳߴ磬ͬ��paddingBR
//Ϊ��Ҫ��imgͼ�����½������ĳߴ硣
void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
                                    Size paddingTL, Size paddingBR) const
{
    //�ú���ֻ�ܼ���8λ������ȵĵ�ͨ������3ͨ��ͼ��.
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

    //��ͼ�������������������,���ﲻ��Ϊ�˼����Ե�ݶȶ��������䣬��Ϊ
    //Ϊ�˱�Ե�ݶȶ��������ں���Ĵ�����ɵģ���������Ϊʲô������ʱ�������ס�
	//����gradient��ͼ�Ĵ�С,��64*128==��112*160��������5*7=35�����ڣ�windowstride:8��
	//ÿ������105��block,105*36=3780ά��������
	
	//paddingTL.width=16,paddingTL.height=24
    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
                  img.rows + paddingTL.height + paddingBR.height);
	//ע��grad��qangle��2ͨ���ľ���Ϊ3D-trilinear��ֵ�е�orientationά�ȣ�����άΪ����x��y 
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

	Size wholeSize;
    Point roiofs;
    //locateROI�ڴ˴������imgͼ���Ǵ�������ͼ����ĳһ���ֵ����ģ���ô�丸ͼ��
    //�Ĵ�С�ߴ��ΪwholeSize�ˣ�imgͼ�����Ͻ�����ڸ�ͼ���λ�õ��Ϊroiofs�ˡ�
    //�������������丸ͼ�����img�ˣ����������wholeSize�ͺ�img.size()��һ���ģ�
    //��Ӧ����������2�߲�ͬ����Ϊ����Ĺ�ϵ�Ƚϲ��ö�������Ȩ�ҽ�wholesSize���Ϊ
    //img��size������roiofs��Ӧ�����ΪPoint(0, 0)�ˡ�
    img.locateROI(wholeSize, roiofs);
	//img�����һ����ͼ��IMG��Region of interesting,��ôIMG��img�����ڴ�  
    //����IMG(120x120),imgȡ��IMG��һ����TL���꣨10,10����BR���꣨109,109����ô�ߴ�Ϊ��100x100��  
    //��������ͷ��ظ�����IMG��size��120x120�����Լ�img��IMG�е�����ƫ�ƣ�roiofs.x=10,roiofs.y=10�� 
	
	//wholeSizeΪparent matrix��С��������չ��gradsize�Ĵ�С
	//roiofs��Ϊimg��parent matrix�е�ƫ��
	//����������img=parent matrix;�����ڸ�����img�Ǵ�parent img�г�ȡ��10�����λ��
	//����OpenCv��������ô������ʹ��img��parent img����ϵ�����Ǻ��˽�
	//wholeSize��roiofs����paddingʱ���ã����Բ��ܣ�����Ϊ�����img==parent img���Ƿ��Ǵ�parent img��ȡ������ν
    int i, x, y;
    int cn = img.channels();

    //_lutΪ��������������Ϊ��������ֵ�Ĵ洢���ұ�
    Mat_<float> _lut(1, 256);
    const float* lut = &_lut(0,0);//ֻ�ܶ�  

    //gammaУ��ָ���ǽ�0��256������ֵȫ�������ţ�����Χ��С�ˣ��ұ任��Χ�����������ˣ�
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;

	//���ٿռ��xmap��ymap�����и�ռgradsize.width+2��gradsize.height+2�ռ�
	//+2��Ϊ�˼���dx,dyʱ��[-1,0,1]����,��ʹ������ͼ���У����Ե�����ݶ�ʱ����Ҫ�ٶ����һ�����ص�
    //��������Ϊgradsize.width+gradsize.height+4������buffer
    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2; 

    //����֮��˼borderType�͵���4�ˣ���Ϊopencv��Դ���������¶���ġ�
    //#define IPL_BORDER_REFLECT_101    4
    //enum{...,BORDER_REFLECT_101=IPL_BORDER_REFLECT_101,...}
    //borderTypeΪ�߽��������������ص�ķ�ʽ��   
    /*
    Various border types, image boundaries are denoted with '|'

    * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
    * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
    * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
    * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg        
    * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
   */
    const int borderType = (int)BORDER_REFLECT_101;
	//һ�ֺ���ֵĲ�ֵ��ʽ����չ�����ı�Ե��ԭͼ���е�����ֵ����û��������չ�洢�ռ�  
    //����˵ԭͼΪ 100x100������Ҫ���ʣ�-10��-10����ֵ�������ڴ����治���������ֵ�����ֲ�ֵ����������ԭͼ�����Ҹ����ص㣨���磨5,6������ֵ��Ϊ��-10,-10����ֵ  
    //Ҳ���ǽ���չ������귶Χ���磨120x120��ӳ�䵽��100x100����x,y����ֱ�ӳ�䣬ӳ������xmap,ymap�������������xmap[-10]=5,ymap[-10]=6  
	
	/*int borderInterpolate(int p, int len, int borderType)
      ���в���p��ʾ���������ͼ���һ�����꣬����ڶ�Ӧ����������ԣ�
		  ����len��ʾ��ӦԴͼ���һ��������ĳ��ȣ�
		  ����borderType��ʾΪ�������ͣ��������Ѿ��й�����.
      ������������������Ǵ����������ص������ƶϳ�Դͼ���ж�Ӧ�õ������ֵ��
   */
   
    /*�����xmap��ymapʵ�ʺ�����ʲô�أ���ʵxmap���������ľ���
	  �����ͼ���һ�����ص��Ӧ��ԭͼ��img�е����غ����꣬���Կ�
	  ����xmap��������ЩԪ�ص�ֵ����ͬ�ģ���Ϊ����ͼ��϶����Ӧ
	  ��ԭͼ��img�е�ĳһλ�ã���img����ߴ��ڵ�����Ҳ���Ӧ��λ�á�
	  ͬ��ymap�����������������ͼ���һ�����ص��Ӧ��ԭͼ��img
	  �е����������ꡣ
	*/
    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
                        wholeSize.width, borderType) - roiofs.x;
    
	for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
                        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* dbuf = _dbuf;
    //DXΪˮƽ�ݶ�ͼ��DYΪ��ֱ�ݶ�ͼ��MagΪ�ݶȷ���ͼ��AngleΪ�ݶȽǶ�ͼ
    //�ù��췽���ĵ�4��������ʾ����Mat���������ڴ��д�ŵ�λ�á��ɴ˿���
    //��������4��ͼ�����ڴ����������洢�ġ�
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    int _nbins = nbins;
    //angleScale==9/pi;
    float angleScale = (float)(_nbins/CV_PI);//��ĳһ���ȣ���Ӧ������һ��bin��scale  

    for( y = 0; y < gradsize.height; y++ )
    {
    //imgPtr������ָ����imgͼ��ĵ�y���׵�ַ��prePtrָ����img��y-1���׵�ַ��
    //nextPtrָ����img��y+1���׵�ַ��
        const uchar* imgPtr  = img.data + img.step*ymap[y];
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);
    
    //����ͼ��imgΪ��ͨ��ͼ��ʱ�ļ���
        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
				//imgPtrָ��img��y����Ԫ�أ�imgPtr[x]����ʾ��(x,y)���أ�������ֵλ��0~255����Ӧlut[0]~lut[255]
				//������������Ϊ120�����Ӧlut[120]������gammaУ����lut[120]=sqrt(120)
				//���ڲ������������أ�����imgPtr[-1]�޷���ʾgradsize��-1λ��Ԫ�أ�����Ҫ�и�ת��
				//imgPtr[-1-paddingTL.width+roiofs.x],��imgPtr[xmap[-1]]����gradsize��-1λ��Ԫ��Ϊimg��xmap[-1]λ�õ�Ԫ�� 
				
				//����2���Dx��Dy�ͼ�������ˣ���Ϊ���Ӧ���ڴ涼��dbuf��
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
    //��cn==3ʱ��Ҳ��������ͼ��Ϊ3ͨ��ͼ��ʱ�Ĵ���
	//ȡB,G,Rͨ�����ݶ�ģ�����ݶ���Ϊ�õ���ݶ�
        else
        {
            for( x = 0; x < width; x++ )
            {
        //x1��ʾ��y�е�x1�еĵ�ַ
                int x1 = xmap[x]*3;
                float dx0, dy0, dx, dy, mag0, mag;

        //p2Ϊ��y�е�x+1�еĵ�ַ
        //p0Ϊ��y�е�x-1�еĵ�ַ
                const uchar* p2 = imgPtr + xmap[x+1]*3;
                const uchar* p0 = imgPtr + xmap[x-1]*3;
        
        //�����2ͨ���ķ�ֵ
		//Rͨ�����ݶ�
                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

        //�����1ͨ���ķ�ֵ
		//Gͨ�����ݶ�
                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;

        //ȡ��ֵ�����Ǹ�ͨ��
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

        //�����0ͨ���ķ�ֵ
                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;

        //ȡ��ֵ�����Ǹ�ͨ��
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                //������ˮƽ�ʹ�ֱ�����ϵ��ݶ�ͼ��
				dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

		//cartToPolar()�����Ǽ���2�������ӦԪ�صķ��ȺͽǶȣ�
		//���һ�������Ƕ��Ƿ�ʹ�ö�����ʾ��false��ʾ���ö�����ʾ�����û��ȱ�ʾ��
        cartToPolar( Dx, Dy, Mag, Angle, false );

        for( x = 0; x < width; x++ )
        {
			//������ݶȷ������������ڵ�bin��ģ������ֻ��һ��ģ��������������ֵ��  
            //���Բ�ֵ������ĳ�������Ӧ������ bin 7.6,�������ǵ�bin���������ģ��������룬�������ֵ�bin 8��̫�ֲ���  
            //�ǾͰ��õ㵽bin7,bin8�ľ�����䣬������������8����������7�� 
            //-5<angle<4
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;//ÿһ��pi/9,��������t������һ����Ȼ��t/(pi/9)
            //cvFloor()���ز����ڲ������������
			//hidx={-5,-4,-3,-2,-1,0,1,2,3,4};
            int hidx = cvFloor(angle);
            //0<=angle<1;angle��ʾ����˼���������ڵĽ�С���Ǹ�bin�Ļ��Ⱦ���(�����Ȳ�)
            angle -= hidx;
            //gradPtrΪgradͼ���ָ��
			//gradPtr[x*2]��ʾ������x���ݶȷ������ڽ�С���Ǹ�bin�ķ���Ȩ�أ�
			//gradPtr[x*2+1]��ʾ������x���ݶȷ������ڽϴ���Ǹ�bin�ķ���Ȩ��
			gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;

            if( hidx < 0 )
                hidx += _nbins;
            else if( hidx >= _nbins )
                hidx -= _nbins;
			//����Ƿ�<9
            assert( (unsigned)hidx < (unsigned)_nbins );

			//��������ݶȷ������ڵ���������bin���  
            qanglePtr[x*2] = (uchar)hidx;//Ҳ������ȡ��
            hidx++;
            //-1�ڲ����еı�ʾΪ11111111,��-1����Ļ������Լ������ˣ�
			//0�ڲ����еı�ʾΪ00000000,��0����Ľ������0��.
			//ע�⵽nbins=9ʱ��hidx���ֵֻΪ8 
            hidx &= hidx < _nbins ? -1 : 0;
			
			//qangle��ͨ���ֱ������ڵ�����bin
            qanglePtr[x*2+1] = (uchar)hidx;
        }
    }
}


struct HOGCache
{
    struct BlockData
    {
        BlockData() : histOfs(0), imgOffset() {}
		//��blockΪ��λ��Ʃ��block[0]�е�36��bin���ڴ���λ����ǰ��
		//��block[1]�е�36��bin�洢λ���������ڴ�������һ����������ƫ�ã���ΪhistOfs:hist offset
        int histOfs;

		//imgOffset��ʾ��block�ڼ�ⴰ��window�е�λ��
        Point imgOffset;
    };

	//PixData�����߳����бȽϻ�ɬ�Ĳ��֣����������������
	//gradOfs:��pixel��grad��Mat grad�е�λ�ã���һ������(grad.cols*i+j)*2,2��ʾ2ͨ��
	//qangleOfs:pixel��angle��Mat qangle�е�λ�ã���һ������(qangle.cols*i+j)*2,2��ʾ2ͨ��
	//histOfs[4]:�ں�������У����߰�һ��block�е����ط�Ϊ�ĸ�����ÿ����������������ĸ���ͬCell�е�hist�й���
	//��һ�������н���ֱ��ͼͳ�ƣ����������ĸ�Cell�Ĳ�ֱͬ��ͼ��histOfs[i]��ʾÿ�������еĵ�i��ֱ��ͼ
	//������blockֱ��ͼ�洢�ռ��еľ���ԭʼλ�õ�ƫ��
	//��Ȼ��һ��Cell��hist���Ӧ��histOfs[0]=0,���������У�histOfs[1]=9,histOfs[2]=18,histOfs[3]=27
	//|_1_|_2_|_3_|_4_|һ��block�ĸ�cell�������ÿ��cell�ַ��ķ֣�1,2,5,6������ͳ������hist[0],3,4,7,8��hist[1]...
	//|_5_|_6_|_7_|_8_|���߽�һ��block��Ϊ���Ŀ�����Ϊ��A��1,4,13,16/B��2,3,14,15/C��5,9,8,12/D��6,7,10,11
	//|_9_|_10|_11|_12|������ΪA�����е�����ֻ����������Cell�е�hist�й��ף��������������ֻ�����һ��hist
	//|_13|_14|_15|_16|��B����2,3�����ػ��Cell0��Cell1�е�hist�й��ף���Ӧ�Ļ����hist[0]��hist[1],14,15����
	//C������B�������ƣ������������Cell��hist����Ӱ�죬��D�����������ĸ�Cell��hist����Ӱ��
	//histWeights��ÿ�����ضԲ�ͬcell��hist���״�С����������block�е�λ�þ���
	//���˾�������������trilinear��ֵ�ж���position��x��y����ά�ȵĲ�ֵ
	//�������صĽǶȶ�����������bin��Ȩ����HOGDescriptor::computerGradient���������֣�����trilinear���
	//��ʵ������Ϊÿ�����ض�������cell��hist��Ӱ�죬���С������ؾ����cell���ĵľ������
	//Ʃ�紦�����ĵ����أ�8,8��������Ϊ��ÿ��cell��hist����һ�������������Ȩ�صķ���Ҳ���Կ���
	//gradWeight��Ϊ��ֵ���˹Ȩ�صĳ˻�
	//���и�˹Ȩ��ѡ��exp^(-(dx^2+dy^2)/��2*sigma^2��),sigma��HOGDescriptor�о���,��block��(8,8)Ϊ����
	//����gradWeight��histWeight��gradWeight��Ϊ��ͬһ��Cell�в�ͬԪ�ض�hist�Ĺ����ǲ�һ���ģ��ɶ�ά��˹�ֲ�����
	//��histweight˵����һ��Ԫ�ضԲ�ͬcell�е�hist�Ĺ��ײ�ͬ���乱����������������cell�ľ������
	
	//����
	//|_A_|_B_|_C_|_D_|
	//|_E_|_F_|_G_|_H_|
	//|_I_|_J_|_K_|_L_|
	//|_M_|_N_|_O_|_P_|
    struct PixData
    {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    HOGCache();
    HOGCache(const HOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);
    virtual ~HOGCache() {};
    virtual void init(const HOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);

	//windowsInImage����Image�к����ɲ������ٸ�windows
    Size windowsInImage(Size imageSize, Size winStride) const;
	//����img��С�������ƶ���������������ŵõ�������img�е�λ��
    Rect getWindow(Size imageSize, Size winStride, int idx) const;

	//bufΪ�洢blockdata���ڴ�ռ䣬ptΪblock��parent img�е�λ��
    const float* getBlock(Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;

    vector<PixData> pixData;
    vector<BlockData> blockData;

	//���µĲ�����Ϊ�˳�������ص���block��Ϣ�������ص���block��Ϣ�ظ�������õ�һ�ֻ���˼������������� 
    bool useCache;//�Ƿ�洢�Ѿ������block��Ϣ
    vector<int> ymaxCached;//������
    Size winSize, cacheStride;//cacheStride��Ϊ����blockStride,���ʹ���ĸ�����
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;//img����չ��ͼ����imgԭ�������չ��ԭ��ƫ��
    Mat_<float> blockCache;//�����ͼ�����Լ�ⴰ�ڽ��к���ɨ��,��ɨ���block��Ϣ�洢��blockCache�� 
    Mat_<uchar> blockCacheFlags;//�жϵ�ǰblock����ϢblockCache���Ƿ��д洢,1���洢,����ֱ�ӵ��ã�0��δ�洢,��Ҫ����Ϣ�洢��blockCache�� 

    Mat grad, qangle;
    const HOGDescriptor* descriptor;
};

//Ĭ�ϵĹ��캯��,��ʹ��cache,���ֱ��ͼ������СΪ0��
HOGCache::HOGCache()
{
    useCache = false;
    blockHistogramSize = count1 = count2 = count4 = 0;
    descriptor = 0;
}

//���εĳ�ʼ�������������ڲ���init�������г�ʼ��

HOGCache::HOGCache(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

//HOGCache�ṹ��ĳ�ʼ������
//��ʼ����Ҫ������
//1��block�и����ض�block�ĸ�bin�Ĺ���Ȩ�أ��Լ��ڴ洢�ռ��е�λ�� ��¼
//2��block�ĳ�ʼ�����Լ�ÿ��block�ڴ洢�ռ��е�ƫ�ü��ڼ�ⴰ���е�λ�� ��¼
//3�����������ĸ�ֵ
//��û��ʵ�ʼ���HOG
void HOGCache::init(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;

	/*--------------------------------------�����ݶ�----------------------------------------------*/  
    //����ֵ  
    //size��img.cols + paddingTL.width + paddingBR.width,img.rows + paddingTL.height + paddingBR.height,���� CV_32FC2  
    //grad���ݶȵ�ģ�����ݶȷ������ڵ�����bin�Ĳ�ֵֵ  
    //qangle�����ݶȷ������ڵ�����bin�ı��  
	
    //���ȵ���computeGradient()������������ͼ���Ȩֵ�ݶȷ���ͼ�ͽǶ�����ͼ
    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
    //imgoffset��Point���ͣ���_paddingTL��Size���ͣ���Ȼ���Ͳ�ͬ������2�߶���
    //һ����ά���꣬��������opencv��������ֱ�Ӹ�ֵ�ġ�
    imgoffset = _paddingTL;//16,24

    winSize = descriptor->winSize;//64*128
    Size blockSize = descriptor->blockSize;//16*16
    Size blockStride = descriptor->blockStride;//8*8
    Size cellSize = descriptor->cellSize;//8*8
    int i, j, nbins = descriptor->nbins;//9
    
	//rawBlockSizeΪblock�а������ص�ĸ���
    int rawBlockSize = blockSize.width*blockSize.height;//16*16=256
    
    //nblocksΪSize���ͣ��䳤�Ϳ�ֱ��ʾһ��������ˮƽ����ʹ�ֱ������block��
    //����(��Ҫ����block�ڴ����е��ƶ�)
	//�����㷨�ǳ�ֱ�ۣ�Ҳ�������ÿ��Ժ�����һ��ֱ�ӳ������ǵ�(winSize.height - blockSize.height) % blockStride.height ��Ϊ0ʱ���Ͳ�һ��  
    //���� blockSize=4,blockStride=3,winSize.width =9,��ôֱ�ӳ�9/3=3������ֻ��������block, 4|3|2,ֻ���ƶ�һ�� 
    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
                   (winSize.height - blockSize.height)/blockStride.height + 1);//7*15
    
	//ncellsҲ��Size���ͣ��䳤�Ϳ�ֱ��ʾһ��block��ˮƽ����ʹ�ֱ����������
    //��cell����
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);//2*2
    //blockHistogramSize��ʾһ��block�й��׸�hog�����������ĳ���
    blockHistogramSize = ncells.width*ncells.height*nbins;//9*2*2

	
	//����ѵ��ʱ,�öδ��벻�����ã����ڼ��ʱ,�öδ��������������ٶȡ�
	//��ѵ��ʱ,����������С�����ڼ�ⴰ�ڴ�С,�������Ҫ����洢
	//�����ڼ��ʱ���ڴ����ͼ����ڼ�ⴰ��,�������ⴰ���ƶ�ʱ,������ڼ�ⴰ�ھ��д�����ͬ��block��Ϣ
	//Ϊ�˽�ʡʱ��,����֮ǰ�������block��Ϣ,����ֻ��Ҫ����,������δ�������block��Ϣ,�����¼��㲢�洢
	//�����˼·���£���������ͼ��640*480,��ⴰ��Ϊ144*144			//�����������С�����������ٶȣ�����������������
	//�����ͼ��ˮƽ������79��block,��ⴰ�ڴ�ֱ������17��block
	//���������´���֪����blockCacheΪ18*��79*36��=18*2844,blockCacheFlagsΪ17*79,ymxcCachedΪ17
	//�����ϽǴ����ⴰ��λ��,��λ�ڣ�0,0��ʱ,��һ�μ���block��Ϣ,blockCache����û�б����κ���Ϣ�ġ�
	//��λ�ڣ�0,0��ʱ����㣨Ҳ��block���ϽǴ���blockλ�ã���
	//(0,0)---->(128,0) ��Ϣ���洢��blockCache��,�ֱ�ΪblockCache[0][0]--->blockCache[0][17*36],��ӦblockCacheFlags��1
	//(0,128)-->(128,128) blockCache[17][0]-->blockCache[17][17*36]
	//����ⴰ���ƶ�����8,0��ʱ,���Է��������������д�����Ϣ���ظ���,���ǿ���ֱ�ӵ���blockCache�����block��Ϣ
	//���ѣ�136,0��-->(136,128)�����е�block��Ϣ�ӵ�blockCache��,ͬʱ����blockCacheFlags
	//һֱ�������Ƶ�(624,0)���뵽��һ��(0,8),�������̳���,����blockCache��ǰ17�д洢�˴����ͼ����ǰ17*79��block��Ϣ
	//����ⴰ���ƶ���(624,0)ʱ��ʱblockCache�Ѿ��洢����
	//����ⴰ���ƶ���(0,8)ʱ,��18�е���Ϣ��ô�����أ�
	//��ʱ���Ҫ������ǵ�1�е�block��Ϣ�Ѿ�û������,���ǿ��Խ���18�е���Ϣ�����1�е���Ϣ��
	//����ⴰ�ڲ��Ϻ���ɨ��ʱ,����һ�е���Ϣ���ǻ�������һ�е���Ϣ,��˷���,�ﵽ��������ٶȵ�Ŀ��
	//������Ҫ�ᵽһ����ǵ�block��pt=(x,y)=(0,0)-->(624,0)--->(0,128)---->(624.128)
	//������x/cacheStride=blockStride--->Canche_X,y/blockStride--->Cache_Y
	//�Ӷ���blockCache��ȡ����Ӧ��blockCache[Cache_Y][Cache_X*36]
	//��pt��y>128ʱ,��Ӧ�ĵ�18����Ϣ�洢�ڵ�blockCache�еĵ�0��
	//�������ǿ�����ȡ��İ취,y/blockStride%18--->Cache_Y,��Cache_X�ļ��㲻��
	//getblock�����д������ǰ��÷������в����� 
    if( useCache )
    {
		//HOGCache��grad��qangle��discriptor->computerGradient�õ�
        //cacheStride= _cacheStride,�����С���ɲ��������,��ʾ���Ǵ����ƶ��Ĵ�С
        //cacheSize���Ϳ��ʾ������ͼ��cache�У�block��ˮƽ����ʹ�ֱ������ֵĸ���
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
                       (winSize.height/cacheStride.height)+1);
        //blockCacheΪһ��float�͵�Mat��ע����������ֵ
        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        //blockCacheFlagsΪһ��uchar�͵�Mat
        blockCacheFlags.create(cacheSize);
        size_t cacheRows = blockCache.rows;
        //ymaxCachedΪvector<int>����
        //Mat::resize()Ϊ�����һ��������ֻ�Ǹı������������뵥����resize()��������ͬ��
        ymaxCached.resize(cacheRows);
        //ymaxCached�����ڲ�ȫ����ʼ��Ϊ-1
        for(size_t ii = 0; ii < cacheRows; ii++ )
            ymaxCached[ii] = -1;
    }
    
    //weightsΪһ���ߴ�ΪblockSize�Ķ�ά��˹��,����Ĵ�����Ǽ����ά��˹��ϵ��
	//sigmaĬ��ֵΪ4
    Mat_<float> weights(blockSize);//16*16 ��˹ģ��  
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

	//Ȩ�صĶ�ά��˹�ֲ�
    for(i = 0; i < blockSize.height; i++)
        for(j = 0; j < blockSize.width; j++)
        {
            float di = i - blockSize.height*0.5f;
            float dj = j - blockSize.width*0.5f;
            weights(i,j) = std::exp(-(di*di + dj*dj)*scale);
        }

    //vector<BlockData> blockData;��BlockDataΪHOGCache��һ���ṹ���Ա
    //nblocks.width*nblocks.height��ʾһ����ⴰ����block�ĸ�����
    //��cacheSize.width*cacheSize.heigh��ʾһ���Ѿ������ͼƬ�е�block�ĸ���
    blockData.resize(nblocks.width*nblocks.height);//105��block
    //vector<PixData> pixData;ͬ��PixdataҲΪHOGCache�е�һ���ṹ���Ա
    //rawBlockSize��ʾÿ��block�����ص�ĸ���
    //resize��ʾ����ת����������
    pixData.resize(rawBlockSize*3);//256*3(ͨ����)

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //count1,count2,count4�ֱ��ʾblock��ͬʱ��1��cell��2��cell��4��cell�й��׵����ص�ĸ�����
    //�����ò��ұ�ķ��������㡣����ʵ��ʱ����ִ��HoGCache�ĳ�ʼ������Init()
	//������ұ�Ȼ����getWindow()��getBlock()��������ʵ�ֵı�Ĳ���

	count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )//16,��ˮƽ���ٴ�ֱ  
        for( i = 0; i < blockSize.height; i++ )//16
        {
            PixData* data = 0;
            //cellX��cellY��ʾ����block�ڸ����ص����ڵ�cell���������������������С������ʽ���ڡ�
            //ȷ��cell��block�е�λ��
			float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            //cvRound������ӽ�����������;cvFloor���ز����ڲ���������;cvCeil���ز�С�ڲ���������
            //icellX0��icellY0��ʾ����cell��������������ֵΪ�����ص�����cell���Ǹ���С��cell����
            //��Ȼ�˴���������������ʽ�����ˡ�
            //����Ĭ�ϵ�ϵ���Ļ���icellX0��icellY0ֻ����ȡֵ-1,0,1,�ҵ�i��j<3.5ʱ��Ӧ��ֵ��ȡ-1
            //��i��j>11.5ʱȡֵΪ1������ʱ��ȡֵΪ0(ע��i��j�����15����0��ʼ��)
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            //�˴���cellx��celly��ʾ������ʵ����ֵ�������cell����ֵ֮��Ĳ
            //Ϊ�������ͬһ���ضԲ�ͬcell�е�histȨ�صļ��㡣
            cellX -= icellX0;
            cellY -= icellY0;
      
			//ע�⵽unsigned����icellX0=-1ʱ����unsigned��icellX0>2
			//(0~3,0~3)+(0~3,12~15)+(12~15,0~3)+(12~15,12~15)
			//(icellX0,icellY0,icellX1,icellY1)=(-1,-1,0,0),(-1,1,0,2),(1,-1,0,2),(1,1,2,2)===������4
			//(4~11,4~11)==����0,0,1,1��==������1
			//(0~3,4~11)+(12~15,4~11)==��(-1,0,0,1)==������3
			//(4~11,0~3)+(4~11,12~15)==��(0,-1,1,0)==������2
			//���2,3��Ԫ�ض�����cell�е�hist�й���
			//(0~3,4~11):histofs=(0,9,0,0);(12~15,4~11):histofs=(18,27,0,0)
			//(4~11,0~3):histofs=(0,18,0,0);(4~11,12~15):hisofs=(9,27,0,0)
			//���1�У�Ԫ�ض�4��cell��hist�й���,�����4��hist��histofs,����Ϊ(0,9,18,27)
			//���4�У�Ԫ������һ��cell,��ֻ��һ��hist����Ӧ��ֻ��һ��histofs:hist offset
			//�ֱ�ӦΪ��(0,0,0,0),(9,0,0,0),(18,0,0,0),(27,0,0,0)
			//����Ȩ�ص���⿴�����ע�ͣ�ѡ��ڶ������������������
            
			//�������if����˵��icellX0ֻ��Ϊ0,Ҳ����˵block��������(3.5,11.5)֮��ʱ
            //�ж�����ʱ�ر�С�ģ�int ת���� unsigned,(unsigned)(-1)=2^32-1���������������  
			if( (unsigned)icellX0 < (unsigned)ncells.width &&
                (unsigned)icellX1 < (unsigned)ncells.width )
            {
               //�������if����˵��icellY0ֻ��Ϊ0,Ҳ����˵block��������(3.5,11.5)֮��ʱ
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    //ͬʱ��������2��if�������ض�4��cell����Ȩֵ����
                    //rawBlockSize��ʾ����1��block�д洢���ص�ĸ���
                    //��pixData�ĳߴ��СΪblock�����ص��3�����䶨�����£�pixData.resize(rawBlockSize*3);
                    //pixData��ǰ��block���ش�С���ڴ�Ϊ�洢ֻ��block��һ��cell�й��׵�pixel��
					//�м�block���ش�С���ڴ�洢��block��ͬʱ2��cell�й��׵�pixel��
					//������Ϊ��block��ͬʱ4��cell���й��׵�pixel
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    //���������Ľ��Ϊ0
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;//cell 0 ������block��bin�е�ƫ��
                     //Ϊ�����ص��cell0��Ȩ��
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);//���Գ����ĵġ����롱��cell 3 
                    //���������Ľ��Ϊ18
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;//cell 1��ƫ�� 2*9  
                    data->histWeights[1] = cellX*(1.f - cellY);//���Գ����ĵġ����롱�� cell 2 
                    //���������Ľ��Ϊ9
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;//cell 2��ƫ�� 1*9  
                    data->histWeights[2] = (1.f - cellX)*cellY; //���Գ����ĵġ����롱�� cell 1  
                    //���������Ľ��Ϊ27
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;//cell 3��ƫ��3*9  
                    data->histWeights[3] = cellX*cellY;//���Գ����ĵġ����롱�� cell 0  
                }
                else
                //�������else����˵��icellY0ȡ-1����1,Ҳ����˵block��������(0, 3.5)
                //��(11.5, 15)֮��.
                //��ʱ�����ص�����ڵ�2��cell��Ȩ�ع���
                {
                    data = &pixData[rawBlockSize + (count2++)];                    
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        //(unsigned)-1����127>2�����Դ˴�����if����ʱicellY0==1��
                        //icellY1==1;
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    //������if����ʱ��icellY0==-1;icellY1==0;
                    //��Ȼ�ˣ���2�������icellX0==0;icellX1==1;
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            //��block�к�����������(0, 3.5)��(11.5, 15)��Χ��ʱ����
            //icellX0==-1��==1
            else
            {
                
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    //icellX1=icllX0=1;
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }
                //��icllY0=0ʱ����ʱ��2��cell�й���
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {                    
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = cellX*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
                else
                //��ʱֻ�������cell�й���
                {
                    data = &pixData[count1++];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = cellX*cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            //Ϊʲôÿ��block��i,jλ�õ�gradOfs��qangleOfs����ͬ�������µļ��㹫ʽ�أ�
            //������Ϊ�����_img�������Ǵ�������ͼƬ���Ǽ�ⴰ�ڴ�С��ͼƬ������ÿ��
            //��ⴰ���й���block����Ϣ���Կ�������ͬ��
            data->gradOfs = (grad.cols*i + j)*2;//block���ڵģ�0,0��λ�������������ͼ���ƫ�ƣ���ƫ��Ϊ�����block(0,0)��ƫ�� 
            data->qangleOfs = (qangle.cols*i + j)*2;//���㷽ʽ�ܹŹ֣������㻭��ͼ�������ˣ�grad.cols*i�����==+j����ģ���ʵ���� block���ڵģ�0,0����offset���ϴ�offset�Ϳ���ֱ����grad���ҵ���Ӧ���ݶ�  
            //ÿ��block��i��jλ�õ�Ȩ�ض��ǹ̶���
            data->gradWeight = weights(i,j);//�õ�ĸ�˹Ȩֵ����С�뵽block���ĵľ���ɷ���  
        }

    //��֤���еĵ㶼��ɨ����һ��
    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData,��Ƭ����
    //���ݺϲ�  xxx.........yyy.........zzz.........->xxxyyyzzz..................  
    //��.��ʾδ��ֵ�ռ䣬xΪcount1�洢�����ݣ�yΪcount2�洢������...��
    //��pixData�а����ڴ�������������ʡ��2/3���ڴ�
	//�ڴ��д洢˳��Ϊ��1,4,13,16/2,3,5,8,9,12,14,15/6,7,10,11�������ص���Ϣ
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    //��ʱcount2��ʾ�����2��cell�й��׵��������ص�ĸ���
    count2 += count1;
    //��ʱcount4��ʾ�����4��cell�й��׵��������ص�ĸ���
    count4 += count2;

    //�����ǳ�ʼ��pixData,���濪ʼ��ʼ��blockData
    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            //histOfs��ʾ��block�Լ�ⴰ�ڹ��׵�hog�����������������
            //�����е�����
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            //imgOffset��ʾ��block�����Ͻ��ڼ�ⴰ���е�����
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
        //һ����ⴰ�ڶ�Ӧһ��blockData�ڴ棬һ��block��Ӧһ��pixData�ڴ档
}


//buf:�洢�ռ�
//pt:block��parent img�е����꣬��ƫ�ã����Ͻǣ�
//ֻ��ȡһ��block�е���Ϣ����256�����ص�grad��angle��Ϣ��Ϊ36��bin����Ϣ������
//ptΪ��block���Ͻ��ڻ��������е����꣬bufΪָ���ⴰ����blocData��ָ��
//��������һ��block�����ӵ�ָ��
const float* HOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
               (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

    if( useCache )	//Ĭ��δʹ��
    {
        //cacheStride������Ϊ��blockStride��һ����
        //��֤����ȡ��HOGCache����������Ҫ�ģ�����block�ƶ������л����
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        //cacheIdx��ʾ����block����������
        Point cacheIdx(pt.x/cacheStride.width,
                      (pt.y/cacheStride.height) % blockCache.rows);
        //ymaxCached�ĳ���Ϊһ����ⴰ�ڴ�ֱ���������ɵ�block����
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            //ȡ��blockCacheFlags�ĵ�cacheIdx.y�в��Ҹ�ֵΪ0
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        //blockHistָ��õ��Ӧblock�����׵�hog��������������ʼֵΪ��
        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;//64,128,256  
    //pt.x*2������2ͨ������¼block���ϽǶ�Ӧ��grad.data��qangle.data�е�λ��
    const float* gradPtr = (const float*)(grad.data + grad.step*pt.y) + pt.x*2;//block(0,0)�������ݶȷ������ڵ�����bin�ϵĲ�ֵ����  
    const uchar* qanglePtr = qangle.data + qangle.step*pt.y + pt.x*2;//��block(0,0)�ݶȷ������ڵ�����bin��bin���

    CV_Assert( blockHist != 0 );

	//blockHistogramSize=36
    for( k = 0; k < blockHistogramSize; k++ )
        blockHist[k] = 0.f;

	//����һ��block����������256����������Ϊ��λȡ
	//һ�����ذ�����gradofs,qangleofs,gradweight,histofs[4],histweight[4]
	//pixData����256��Ԫ�أ�blockData����105��block
    const PixData* _pixData = &pixData[0];//pixData��init���Ѿ�������ˣ������block��0,0����ƫ��

    //C1��ʾֻ���Լ�����cell�й��׵ĵ�ĸ���
	//ADMP����
    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        //a��ʾ���Ƿ���ָ��
        const float* a = gradPtr + pk.gradOfs;//gradPtr��ʼ��ַ���ɲ�ͬ����Point pt���仯��pk.gradOfsƫ�� 
        float w = pk.gradWeight*pk.histWeights[0];
        //h��ʾ������λָ��
        const uchar* h = qanglePtr + pk.qangleOfs;

        //������2��ͨ������Ϊÿ�����ص�ķ�ֵ���ֽ⵽�������ڵ�����bin����
        //��λ��2��ͨ������Ϊÿ�����ص����λ�����ڴ����е�2��bin�����
        int h0 = h[0], h1 = h[1];//h[0]Ϊangle����bin��λ��0~8��hist[h0]��ʾ��h0��bin���д洢������Ӧ�ķ�����Ȩ��
        float* hist = blockHist + pk.histOfs[0];//blockHistΪbuff�ĵ�ַ��histOfs��Ϊƫ�� 
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        //hist�зŵ�Ϊ��Ȩ���ݶ�ֵ
        hist[h0] = t0; hist[h1] = t1;
    }
	//����
	//BCEINPHL
    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        //��Ϊ��ʱ�����ض�2��cell�й��ף���������һ��cell�Ĺ���
        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        //��һ��cell�Ĺ���
        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    //�ĸ�
	//FGJK����
    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight*pk.histWeights[2];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight*pk.histWeights[3];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    normalizeBlockHistogram(blockHist);

    return blockHist;
}

//L2HysThreshold����L2��һ�������������е�ֵ�ķ�Χ��0,0.2����������L2��һ��
void HOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];

    size_t i, sz = blockHistogramSize;

    float sum = 0;

    //��һ�ι�һ�������ƽ����
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];

    //��ĸΪƽ���Ϳ�����+0.1
    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;

    for( i = 0, sum = 0; i < sz; i++ )
    {
        //��2�ι�һ�����ڵ�1�εĻ����ϼ�����ƽ�ͺ�
        hist[i] = std::min(hist[i]*scale, thresh);//�������ֵΪ0.2
        sum += hist[i]*hist[i];
    }

	//�ڹ�һ��һ�飬ʹ�ø���ƽ����Ϊ1������λ��
    scale = 1.f/(std::sqrt(sum)+1e-3f);

    //���չ�һ�����
    for( i = 0; i < sz; i++ )
        hist[i] *= scale;

}


//���ز���ͼƬ��ˮƽ����ʹ�ֱ�����ж��ٸ���ⴰ��
Size HOGCache::windowsInImage(Size imageSize, Size winStride) const
{
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
                (imageSize.height - winSize.height)/winStride.height + 1);
}

//����img��С�������ƶ���������������ŵõ�������img�е�λ��
//����ͼƬ�Ĵ�С���Ѿ���ⴰ�ڻ����Ĵ�С�Ͳ���ͼƬ�еļ�ⴰ�ڵ��������õ���������
//��ⴰ�ڵĳߴ磬����������Ϣ
Rect HOGCache::getWindow(Size imageSize, Size winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;//��
    int x = idx - nwindowsX*y;//����
    return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}

//img������������ͼ��
//descriptors:Hog�����ṹ
//winStride:�����ƶ�����
//padding������ͼ����سߴ�
//locations:��������������ֱ��ȡ(0,0),������Ϊ��������������귶Χ�ڵĵ�����
void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
                            Size winStride, Size padding,
                            const vector<Point>& locations) const
{
    //Size()��ʾ���Ϳ���0
	//��winStride.width=0,winStride.height=0��ȡ(8,8)
    if( winStride == Size() )
        winStride = cellSize;
    //gcdΪ�����Լ�����������Ĭ��ֵ�Ļ�����2����ͬ
	//alignSize(size_t sz, int n)
	//����n�ı����в�С��sz����С������padding.width��������
	//��Ĭ�ϲ�����cacheStride=blockStride=(8,8)��padding.width=24,padding.height=16,����Ҳ����Ҫ�������ɺ��� 
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));
	//������ֻ��һ�����ڣ����δ����
	//����������������˵���������10��ͼ����δ���������10������
    size_t nwindows = locations.size();
    //alignSize(m, n)����n�ı������ڵ���m����Сֵ
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	//��nwidows=0ʱ����ͼ��֮���ټ��㹲�ж��ٴ���area()=size.width*size.height,windowsInImage���ص���nwidth��nheight
	//�ڼ��ʱ�����ã����ڼ��ʱ�ǲ�֪��Ҫ�����Ŀ�����ģ�������Ҫ������ͼ����Ҫ���ٴ���
	//ѵ��ʱ����������С��Ϊ���ڴ�С,���Բ���Ҫ����洢block��Ϣ,��useCache=0,nwindows=1;
	//���ʱ���ڴ����ͼ����ڼ�ⴰ�ڴ�С,������Ҫ����洢�ظ���block��Ϣ,��useCache=1,��Ҫ���¼���nwindows
	//detect�����е�useCacheĬ��ֵΪ1,�����ʱ����Ҫ����洢block��Ϣ��
	//compute�����е�useCacheĬ��ֵΪ0,detect�����compute,��ı�useCache��ֵ
    if( !nwindows )
        //Mat::area()��ʾΪMat�����
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();//һ��hog���������ȡ�һ������������������С��2*2*9*15*7=3780
    
	//resize()Ϊ�ı�����������������پ����������ֻ�������ٺ��
    //��Щ�У�����������������������е��С�
    //���ｫ�����ӳ�����չ������ͼƬ
    descriptors.resize(dsize*nwindows);//ע�⵽�㷨��������СΪ64*128����ʵ������������ģ�ʵ������������Ҫ����nwindows

	//descriptor�洢��nwindows�Σ�ÿ���ַ�nblocks=105�Σ�ÿ������36��bi
    for( size_t i = 0; i < nwindows; i++ )
    {
        //descriptorΪ��i����ⴰ�ڵ���������λ�á�
        float* descriptor = &descriptors[i*dsize];
       
        Point pt0;
		//locations.empty()Ϊ�շ���1
        //�ǿ�
        if( !locations.empty() )
        {
            pt0 = locations[i];
            //�Ƿ��ĵ�
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        //locationsΪ��
        else
        {
            //pt0Ϊû������ǰͼ���Ӧ�ĵ�i����ⴰ��
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
			//imgOffset = Point(j*blockStride.width,i*blockStride.height),block��window�е�λ��
			//pt0:Ϊimg��parent img�е�λ�ã�ע�⵽getBlock(pt,dst)��pt����ָ����parent img�е�λ��
            //ptΪblock�����Ͻ���Լ��ͼƬ������
            Point pt = pt0 + bj.imgOffset;

			//histOfs=(j*nblocks.height + i)*blockHistogramSize,nblocks.height=15
            //dstΪ��block����������ͼƬ�������ӵ�λ��
            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if( src != dst )
                for( int k = 0; k < blockHistogramSize; k++ )
                    dst[k] = src[k];
        }
    }
}


void HOGDescriptor::detect(const Mat& img,
    vector<Point>& hits, vector<double>& weights, double hitThreshold, 
    Size winStride, Size padding, const vector<Point>& locations) const
{
    //hits�������Ƿ��ϼ�⵽Ŀ��Ĵ��ڵ����ϽǶ�������
    hits.clear();
    if( svmDetector.empty() )
        return;

    if( winStride == Size() )//δָ��winStride������£�winStride==��8,8��  
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));// gcd,�����Լ����Ĭ�Ͻ����8,8��  
    size_t nwindows = locations.size();// Ĭ�ϣ�0  
	//���������Լ��趨��LTpading=BRpading=pading,���е���ʹ��pading�Ŀ����casheStride�Ŀ�߶��룬������4�ֽڲ������ 
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);//alignSize(m, n)������n�ı����д��ڵ���m����Сֵ  
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

	/*--------------------------------------------------------------------*/  
    //  1.�����ݶȵ�ģ������  
    //  2.Ԥ�ȼ������һ��block��bin��ƫ�ơ���˹Ȩ�ء���ֵ����  
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	//�����ˣ�computeGradient��������pading����ݶ�  
    //Note:�߶ȱ仯ʱ�����¼������ݶ�  
    //histOfs = (j*nblocks.height + i)*blockHistogramSize;  
    
    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();//����img �ļ�ⴰ����

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();//��ⴰ���ڵ�block��
    int blockHistogramSize = cache.blockHistogramSize;//һ��block��histogram��bin������2*2*9  
    size_t dsize = getDescriptorSize();//һ�����ڵ�����������  

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;// > �������������svm�� �ͷ���ϵ�� C ��Ϊ0  
    vector<float> blockHist(blockHistogramSize);

    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
			//�õ���i����ⴰ����pading֮���ͼ���е����������ȥpading,����geitblock��pt += imgoffset; 
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        //svmVecָ��svmDetector��ǰ���Ǹ�Ԫ��
        const float* svmVec = &svmDetector[0];

        int j, k;

        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];//��ⴰ���е�j��block  
            // .histOfs = (j*nblocks.height + i)*blockHistogramSize;  
            // .imgOffset = Point(j*blockStride.width,i*blockStride.height);  
			Point pt = pt0 + bj.imgOffset;//�õ���i����ⴰ���е�j��block��pading֮���ͼ���е�TL����
            
            //vecΪ����ͼƬpt����block���׵�������ָ��
            const float* vec = cache.getBlock(pt, &blockHist[0]);

			//���㵽���೬ƽ��ľ���
            for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                //const float* svmVec = &svmDetector[0];
                s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] +
                    vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
            for( ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }

        if( s >= hitThreshold )
        {
            hits.push_back(pt0);
            weights.push_back(s);
        }
    }
}

//���ñ�����⵽Ŀ��Ŀ��Ŷȣ���Ȩ��
void HOGDescriptor::detect(const Mat& img, vector<Point>& hits, double hitThreshold, 
                           Size winStride, Size padding, const vector<Point>& locations) const
{
    vector<double> weightsV;
    detect(img, hits, weightsV, hitThreshold, winStride, padding, locations);
}

struct HOGInvoker
{
    HOGInvoker( const HOGDescriptor* _hog, const Mat& _img,
                double _hitThreshold, Size _winStride, Size _padding,
                const double* _levelScale, ConcurrentRectVector* _vec, 
                ConcurrentDoubleVector* _weights=0, ConcurrentDoubleVector* _scales=0 ) 
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
    }

    void operator()( const BlockedRange& range ) const
    {
        int i, i1 = range.begin(), i2 = range.end();
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        //��ԭͼƬ��������
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        vector<Point> locations;
        vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            //smallerImgֻ�ǹ���һ��ָ�룬��û�и�������
            Mat smallerImg(sz, img.type(), smallerImgBuf.data);
            //û�гߴ�����
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            //�гߴ�����
            else
                resize(img, smallerImg, sz);
			//dst���ڴ�ռ䳬��srcʱ��dst�Ŀռ��ǲ��ǲ�û����С�أ�  
            //Ҳ����˵�ǲ������ͷ��ڴ棬�ٰ����µ�size��������,�ӳ����Ͽ�һֱ��ռԭʼ�ڴ�ռ�����𵽼����ڴ������ͷ����ķѵ�ʱ��  
            
			//�ú���ʵ�����ǽ����ص�ֵ����locations��histWeights��
            //����locations�����Ŀ����������Ͻ�����
            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            for( size_t j = 0; j < locations.size(); j++ )
            {
                //����Ŀ������
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                //�������ųߴ�
                if (scales) {
                    scales->push_back(scale);
                }
            }
            //����svm�����Ľ��ֵ
            if (weights && (!hitsWeights.empty()))
            {
                for (size_t j = 0; j < locations.size(); j++)
                {
                    weights->push_back(hitsWeights[j]);
                }
            }        
        }
    }

    const HOGDescriptor* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    //typedef tbb::concurrent_vector<Rect> ConcurrentRectVector;
    ConcurrentRectVector* vec;
    //typedef tbb::concurrent_vector<double> ConcurrentDoubleVector;
    ConcurrentDoubleVector* weights;
    ConcurrentDoubleVector* scales;
};


void HOGDescriptor::detectMultiScale(
    const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
{
    double scale = 1.;
    int levels = 0;

    vector<double> levelScale;
	//Ҫʹ��ⴰ�ڵĳ߶ȱ�������ַ�������һ��ͼ��ߴ粻�䣬�����ⴰ�ڵĴ�С������������������ⴰ�ڲ��䣬��СͼƬ  
    //����ʹ�õ����ǵڶ��ַ���  
	
	//������������������ǽ�ͼ����С������Ϊ�����߶��Ѿ���С�ˣ�ʵ�ʵ�����ֻ����������ߴ磬С�������ߴ�������޷���� 
    //nlevelsĬ�ϵ���64��
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(img.cols/scale) < winSize.width ||	//С��64��߶ȵĳ߶���������ͼ�εĳߴ��scale0�����ģ�
            cvRound(img.rows/scale) < winSize.height || //��ͼ�����ŵ��Ѿ�С�ڼ�ⴰ��ʱ���Ѿ����������ӳ߶���  
            scale0 <= 1 )
            break;
        //ֻ���ǲ���ͼƬ�ߴ�ȼ�ⴰ�ڳߴ������
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);
	
	/*
	std::vector<Rect> allCandidates;
    std::vector<double> tempScales;
    std::vector<double> tempWeights;
    std::vector<double> foundScales;
	*/
	
    ConcurrentRectVector allCandidates;
    ConcurrentDoubleVector tempScales;
    ConcurrentDoubleVector tempWeights;
    vector<double> foundScales;
    
    //TBB���м���	http://blog.csdn.net/zoufeiyy/article/details/1887579  
    parallel_for(BlockedRange(0, (int)levelScale.size()),
                 HOGInvoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &tempWeights, &tempScales));
    
	//��tempScales�е����ݸ��Ƶ�foundScales�У�back_inserter��ָ��ָ��������������ĩβ��������
    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    //������clear()������ָ�Ƴ����������е�����
    foundLocations.clear();
    //����ѡĿ�괰�ڱ�����foundLocations��
    std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    foundWeights.clear();
    //����ѡĿ����Ŷȱ�����foundWeights��
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));

    if ( useMeanshiftGrouping )
    {
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
    }
    else
    {
        //�Ծ��ο���о���
        groupRectangles(foundLocations, (int)finalThreshold, 0.2);
    }
}

//������Ŀ������Ŷ�
void HOGDescriptor::detectMultiScale(const Mat& img, vector<Rect>& foundLocations, 
                                     double hitThreshold, Size winStride, Size padding,
                                     double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
{
    vector<double> foundWeights;
    detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride, 
                     padding, scale0, finalThreshold, useMeanshiftGrouping);
}

typedef RTTIImpl<HOGDescriptor> HOGRTTI;

CvType hog_type( CV_TYPE_NAME_HOG_DESCRIPTOR, HOGRTTI::isInstance,
                 HOGRTTI::release, HOGRTTI::read, HOGRTTI::write, HOGRTTI::clone);

vector<float> HOGDescriptor::getDefaultPeopleDetector()
{
    static const float detector[] = {
       0.05359386f, -0.14721455f, -0.05532170f, 0.05077307f,
       0.11547081f, -0.04268804f, 0.04635834f, ........
  };
       //����detector����Ĵ�ͷ��β���ɵ�����
    return vector<float>(detector, detector + sizeof(detector)/sizeof(detector[0]));
}
//This function renurn 1981 SVM coeffs obtained from daimler's base. 
//To use these coeffs the detection window size should be (48,96)
vector<float> HOGDescriptor::getDaimlerPeopleDetector()
{
    static const float detector[] = {
        0.294350f, -0.098796f, -0.129522f, 0.078753f,
        0.387527f, 0.261529f, 0.145939f, 0.061520f,
      ........
        };
        //����detector����β���ɵ�����
        return vector<float>(detector, detector + sizeof(detector)/sizeof(detector[0]));
}
}