#include "precomp.hpp"
#include <iterator>
#ifdef HAVE_IPP
#include "ipp.h"
#endif

/*
Size winSize;//窗口大小
Size blockSize;//Block大小
Size blockStride;//block每次移动宽度包括水平和垂直两个方向
Size cellSize;//Cell单元大小
int nbins;//直方图bin数目
int derivAperture;//不知道什么用		?????????????????
double winSigma;//高斯函数的方差
int histogramNormType;//直方图归一化类型，具体见论文
double L2HysThreshold;//L2Hys化中限制最大值为0.2 
bool gammaCorrection;//是否Gamma校正 
vector<float> svmDetector;//检测算子
*/
namespace cv
{

size_t HOGDescriptor::getDescriptorSize() const
{
	//检测数据的合理性
    //下面2个语句是保证block中有整数个cell;保证block在窗口中能移动整数次
    CV_Assert(blockSize.width % cellSize.width == 0 &&
        blockSize.height % cellSize.height == 0);
    CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
        (winSize.height - blockSize.height) % blockStride.height == 0 );
    //返回的nbins是每个窗口中检测到的hog向量的维数
    return (size_t)nbins*
        (blockSize.width/cellSize.width)*
        (blockSize.height/cellSize.height)*
        ((winSize.width - blockSize.width)/blockStride.width + 1)*
        ((winSize.height - blockSize.height)/blockStride.height + 1);
}

//winSigma到底是什么作用呢？
double HOGDescriptor::getWinSigma() const
{
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}

//svmDetector是HOGDescriptor内的一个成员变量，数据类型为向量vector。
//用来保存hog特征用于svm分类时的系数的.
//该函数返回为真的实际含义是什么呢？保证与hog特征长度相同，或者相差1，但为什么
//相差1也可以呢？
bool HOGDescriptor::checkDetectorSize() const
{
    size_t detectorSize = svmDetector.size(), descriptorSize = getDescriptorSize();
    //三种情况任意一种为true则表达式为true，实际上是最后一种
	return detectorSize == 0 ||
        detectorSize == descriptorSize ||
        detectorSize == descriptorSize + 1;
}

void HOGDescriptor::setSVMDetector(InputArray _svmDetector)
{  
    //这里的convertTo函数只是将图像Mat属性更改，比如说通道数，矩阵深度等。
    //这里是将输入的svm系数矩阵全部转换成浮点型。
    _svmDetector.getMat().convertTo(svmDetector, CV_32F);
    CV_Assert( checkDetectorSize() );
}

#define CV_TYPE_NAME_HOG_DESCRIPTOR "opencv-object-detector-hog"

//FileNode是opencv的core中的一个文件存储节点类，这个节点用来存储读取到的每一个文件元素。
//一般是读取XML和YAML格式的文件
//又因为该函数是把文件节点中的内容读取到其类的成员变量中，所以函数后面不能有关键字const
bool HOGDescriptor::read(FileNode& obj)
{
    //isMap()是用来判断这个节点是不是一个映射类型，如果是映射类型，则每个节点都与
    //一个名字对应起来。因此这里的if语句的作用就是需读取的文件node是一个映射类型
    if( !obj.isMap() )
        return false;
    //中括号中的"winSize"是指返回名为winSize的一个节点，因为已经知道这些节点是mapping类型
    //也就是说都有一个对应的名字。
    FileNodeIterator it = obj["winSize"].begin();
    //操作符>>为从节点中读入数据，这里是将it指向的节点数据依次读入winSize.width,winSize.height
    //下面的几条语句功能类似
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
    
    //isSeq()是判断该节点内容是不是一个序列
    FileNode vecNode = obj["SVMDetector"];
    if( vecNode.isSeq() )
    {
        vecNode >> svmDetector;
        CV_Assert(checkDetectorSize());
    }
    //上面的都读取完了后就返回读取成功标志
    return true;
}
    
void HOGDescriptor::write(FileStorage& fs, const String& objName) const
{
    //将objName名字输入到文件fs中
    if( !objName.empty() )
        fs << objName;

    fs << "{" CV_TYPE_NAME_HOG_DESCRIPTOR
    //下面几句依次将hog描述子内的变量输入到文件fs中，且每次输入前都输入
    //一个名字与其对应，因此这些节点是mapping类型。
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
        //svmDetector则是直接输入序列，也有对应的名字。
        fs << "SVMDetector" << "[:" << svmDetector << "]";
    fs << "}";
}

//从给定的文件中读取参数
bool HOGDescriptor::load(const String& filename, const String& objname)
{
    FileStorage fs(filename, FileStorage::READ);
    //一个文件节点有很多叶子，所以一个文件节点包含了很多内容，这里当然是包含的
    //HOGDescriptor需要的各种参数了。
    FileNode obj = !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

//将类中的参数以文件节点的形式写入文件中。
void HOGDescriptor::save(const String& filename, const String& objName) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    write(fs, !objName.empty() ? objName : FileStorage::getDefaultObjectName(filename));
}

//复制HOG描述子到c中
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
    //vector类型也可以用等号赋值
    c.svmDetector = svmDetector;
	c.nlevels = nlevels; 
} 

//img:原始图像
//grad:记录每个像素所属bin对应的权重的矩阵,为幅值乘以权值
//这个权值是关键，也很复杂：包括高斯权重，三次插值的权重，在本函数中先值考虑幅值和相邻bin间的插值权重
//qangle:记录每个像素角度所属的bin序号的矩阵,均为2通道,为了线性插值
//paddingTL:Top和Left扩充像素数
//paddingBR:类似同上
//功能：计算img经扩张后的图像中每个像素的梯度和角度

//计算图像img的梯度幅度图像grad和梯度方向图像qangle.
//paddingTL为需要在原图像img左上角扩增的尺寸，同理paddingBR
//为需要在img图像右下角扩增的尺寸。
void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
                                    Size paddingTL, Size paddingBR) const
{
    //该函数只能计算8位整型深度的单通道或者3通道图像.
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

    //将图像按照输入参数进行扩充,这里不是为了计算边缘梯度而做的扩充，因为
    //为了边缘梯度而扩充是在后面的代码完成的，所以这里为什么扩充暂时还不明白。
	//计算gradient的图的大小,由64*128==》112*160，则会产生5*7=35个窗口（windowstride:8）
	//每个窗口105个block,105*36=3780维特征向量
	
	//paddingTL.width=16,paddingTL.height=24
    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
                  img.rows + paddingTL.height + paddingBR.height);
	//注意grad和qangle是2通道的矩阵，为3D-trilinear插值中的orientation维度，另两维为坐标x与y 
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

	Size wholeSize;
    Point roiofs;
    //locateROI在此处是如果img图像是从其它父图像中某一部分得来的，那么其父图像
    //的大小尺寸就为wholeSize了，img图像左上角相对于父图像的位置点就为roiofs了。
    //对于正样本，其父图像就是img了，所以这里的wholeSize就和img.size()是一样的，
    //对应负样本，这2者不同；因为里面的关系比较不好懂，这里权且将wholesSize理解为
    //img的size，所以roiofs就应当理解为Point(0, 0)了。
    img.locateROI(wholeSize, roiofs);
	//img如果是一个大图像IMG的Region of interesting,那么IMG和img共享内存  
    //比如IMG(120x120),img取自IMG的一部分TL坐标（10,10），BR坐标（109,109）那么尺寸为（100x100）  
    //这个函数就返回父矩阵IMG的size（120x120），以及img在IMG中的坐标偏移（roiofs.x=10,roiofs.y=10） 
	
	//wholeSize为parent matrix大小，不是扩展后gradsize的大小
	//roiofs即为img在parent matrix中的偏置
	//对于正样本img=parent matrix;但对于负样本img是从parent img中抽取的10个随机位置
	//至于OpenCv具体是怎么操作，使得img和parent img相联系，不是很了解
	//wholeSize与roiofs仅在padding时有用，可以不管，就认为传入的img==parent img，是否是从parent img中取出无所谓
    int i, x, y;
    int cn = img.channels();

    //_lut为行向量，用来作为浮点像素值的存储查找表
    Mat_<float> _lut(1, 256);
    const float* lut = &_lut(0,0);//只能读  

    //gamma校正指的是将0～256的像素值全部开根号，即范围缩小了，且变换范围都不成线性了，
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;

	//开辟空间存xmap和ymap，其中各占gradsize.width+2和gradsize.height+2空间
	//+2是为了计算dx,dy时用[-1,0,1]算子,即使在扩充图像中，其边缘计算梯度时还是要再额外加一个像素的
    //创建长度为gradsize.width+gradsize.height+4的整型buffer
    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2; 

    //言外之意思borderType就等于4了，因为opencv的源码中是如下定义的。
    //#define IPL_BORDER_REFLECT_101    4
    //enum{...,BORDER_REFLECT_101=IPL_BORDER_REFLECT_101,...}
    //borderType为边界扩充后所填充像素点的方式。   
    /*
    Various border types, image boundaries are denoted with '|'

    * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
    * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
    * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
    * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg        
    * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
   */
    const int borderType = (int)BORDER_REFLECT_101;
	//一种很奇怪的插值方式，扩展出来的边缘用原图像中的像素值，并没有真正扩展存储空间  
    //比如说原图为 100x100，现在要访问（-10，-10）的值，但是内存里面不不存在这个值，这种插值方法就是在原图像中找个像素点（比如（5,6））的值作为（-10,-10）的值  
    //也就是将扩展后的坐标范围比如（120x120）映射到（100x100）。x,y坐标分别映射，映射表存在xmap,ymap。上面的例子中xmap[-10]=5,ymap[-10]=6  
	
	/*int borderInterpolate(int p, int len, int borderType)
      其中参数p表示的是扩充后图像的一个坐标，相对于对应的坐标轴而言；
		  参数len表示对应源图像的一个坐标轴的长度；
		  参数borderType表示为扩充类型，在上面已经有过介绍.
      所以这个函数的作用是从扩充后的像素点坐标推断出源图像中对应该点的坐标值。
   */
   
    /*这里的xmap和ymap实际含义是什么呢？其实xmap向量里面存的就是
	  扩充后图像第一行像素点对应与原图像img中的像素横坐标，可以看
	  出，xmap向量中有些元素的值是相同的，因为扩充图像肯定会对应
	  到原图像img中的某一位置，而img本身尺寸内的像素也会对应该位置。
	  同理，ymap向量里面存的是扩充后图像第一列像素点对应于原图想img
	  中的像素纵坐标。
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
    //DX为水平梯度图，DY为垂直梯度图，Mag为梯度幅度图，Angle为梯度角度图
    //该构造方法的第4个参数表示矩阵Mat的数据在内存中存放的位置。由此可以
    //看出，这4幅图像在内存中是连续存储的。
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    int _nbins = nbins;
    //angleScale==9/pi;
    float angleScale = (float)(_nbins/CV_PI);//算某一弧度，对应落在哪一个bin的scale  

    for( y = 0; y < gradsize.height; y++ )
    {
    //imgPtr在这里指的是img图像的第y行首地址；prePtr指的是img第y-1行首地址；
    //nextPtr指的是img第y+1行首地址；
        const uchar* imgPtr  = img.data + img.step*ymap[y];
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);
    
    //输入图像img为单通道图像时的计算
        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
				//imgPtr指向img第y行首元素，imgPtr[x]即表示第(x,y)像素，其亮度值位于0~255，对应lut[0]~lut[255]
				//即若像素亮度为120，则对应lut[120]，若有gamma校正，lut[120]=sqrt(120)
				//由于补充了虚拟像素，即在imgPtr[-1]无法表示gradsize中-1位置元素，而需要有个转换
				//imgPtr[-1-paddingTL.width+roiofs.x],即imgPtr[xmap[-1]]，即gradsize中-1位置元素为img中xmap[-1]位置的元素 
				
				//下面2句把Dx，Dy就计算出来了，因为其对应的内存都在dbuf中
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
    //当cn==3时，也就是输入图像为3通道图像时的处理。
	//取B,G,R通道中梯度模最大的梯度作为该点的梯度
        else
        {
            for( x = 0; x < width; x++ )
            {
        //x1表示第y行第x1列的地址
                int x1 = xmap[x]*3;
                float dx0, dy0, dx, dy, mag0, mag;

        //p2为第y行第x+1列的地址
        //p0为第y行第x-1列的地址
                const uchar* p2 = imgPtr + xmap[x+1]*3;
                const uchar* p0 = imgPtr + xmap[x-1]*3;
        
        //计算第2通道的幅值
		//R通道的梯度
                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

        //计算第1通道的幅值
		//G通道的梯度
                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;

        //取幅值最大的那个通道
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

        //计算第0通道的幅值
                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;

        //取幅值最大的那个通道
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                //最后求出水平和垂直方向上的梯度图像
				dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

		//cartToPolar()函数是计算2个矩阵对应元素的幅度和角度，
		//最后一个参数角度是否使用度数表示，false表示不用度数表示，即用弧度表示。
        cartToPolar( Dx, Dy, Mag, Angle, false );

        for( x = 0; x < width; x++ )
        {
			//保存该梯度方向在左右相邻的bin的模，本来只有一个模何来的两个？插值！  
            //线性插值，比如某点算出来应该属于 bin 7.6,但是我们的bin都是整数的，四舍五入，把他划分到bin 8又太粗糙了  
            //那就按该点到bin7,bin8的距离分配，这样部分属于8，部分属于7。 
            //-5<angle<4
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;//每一格pi/9,那现在算t落在哪一格自然是t/(pi/9)
            //cvFloor()返回不大于参数的最大整数
			//hidx={-5,-4,-3,-2,-1,0,1,2,3,4};
            int hidx = cvFloor(angle);
            //0<=angle<1;angle表示的意思是与其相邻的较小的那个bin的弧度距离(即弧度差)
            angle -= hidx;
            //gradPtr为grad图像的指针
			//gradPtr[x*2]表示的是与x处梯度方向相邻较小的那个bin的幅度权重；
			//gradPtr[x*2+1]表示的是与x处梯度方向相邻较大的那个bin的幅度权重
			gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;

            if( hidx < 0 )
                hidx += _nbins;
            else if( hidx >= _nbins )
                hidx -= _nbins;
			//检测是否<9
            assert( (unsigned)hidx < (unsigned)_nbins );

			//保存与该梯度方向相邻的左右两个bin编号  
            qanglePtr[x*2] = (uchar)hidx;//也是向下取整
            hidx++;
            //-1在补码中的表示为11111111,与-1相与的话就是自己本身了；
			//0在补码中的表示为00000000,与0相与的结果就是0了.
			//注意到nbins=9时，hidx最大值只为8 
            hidx &= hidx < _nbins ? -1 : 0;
			
			//qangle两通道分别存放相邻的两个bin
            qanglePtr[x*2+1] = (uchar)hidx;
        }
    }
}


struct HOGCache
{
    struct BlockData
    {
        BlockData() : histOfs(0), imgOffset() {}
		//以block为单位，譬如block[0]中的36个bin在内存中位于最前面
		//而block[1]中的36个bin存储位置在连续内存中则有一个距离起点的偏置，即为histOfs:hist offset
        int histOfs;

		//imgOffset表示该block在检测窗口window中的位置
        Point imgOffset;
    };

	//PixData是作者程序中比较晦涩的部分，具体见后面程序分析
	//gradOfs:该pixel的grad在Mat grad中的位置，是一个数：(grad.cols*i+j)*2,2表示2通道
	//qangleOfs:pixel的angle在Mat qangle中的位置，是一个数：(qangle.cols*i+j)*2,2表示2通道
	//histOfs[4]:在后面程序中，作者把一个block中的像素分为四个区域，每个区域的像素最多对四个不同Cell中的hist有贡献
	//即一个区域中进行直方图统计，则最多包含四个Cell的不同直方图，histOfs[i]表示每个区域中的第i个直方图
	//在整个block直方图存储空间中的距离原始位置的偏置
	//显然第一个Cell的hist其对应的histOfs[0]=0,依次类推有：histOfs[1]=9,histOfs[2]=18,histOfs[3]=27
	//|_1_|_2_|_3_|_4_|一个block四个cell，这里把每个cell又分四分，1,2,5,6中像素统计属于hist[0],3,4,7,8在hist[1]...
	//|_5_|_6_|_7_|_8_|作者将一个block分为了四块区域为：A：1,4,13,16/B：2,3,14,15/C：5,9,8,12/D：6,7,10,11
	//|_9_|_10|_11|_12|作者认为A区域中的像素只对其所属的Cell中的hist有贡献，即此区域的像素只会产生一个hist
	//|_13|_14|_15|_16|而B区域2,3的像素会对Cell0与Cell1中的hist有贡献，相应的会产生hist[0]与hist[1],14,15类似
	//C区域与B区域类似，会对上下两个Cell的hist产生影响，而D区域会对相邻四个Cell的hist产生影响
	//histWeights：每个像素对不同cell的hist贡献大小，由像素在block中的位置决定
	//个人觉得这是论文中trilinear插值中对于position中x和y两个维度的插值
	//其中像素的角度对于相邻两个bin的权重在HOGDescriptor::computerGradient中已有体现，至此trilinear完成
	//其实作者认为每个像素对于其他cell的hist的影响，其大小与该像素距各个cell中心的距离决定
	//譬如处于中心的像素（8,8）可以认为对每个cell的hist贡献一样，后面程序中权重的分配也可以看出
	//gradWeight：为幅值与高斯权重的乘积
	//其中高斯权重选择exp^(-(dx^2+dy^2)/（2*sigma^2）),sigma在HOGDescriptor中决定,以block中(8,8)为中心
	//区别gradWeight和histWeight，gradWeight认为在同一个Cell中不同元素对hist的贡献是不一样的，由二维高斯分布决定
	//而histweight说的是一个元素对不同cell中的hist的贡献不同，其贡献由其坐标距离各个cell的距离决定
	
	//区域
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

	//windowsInImage返回Image中横竖可产生多少个windows
    Size windowsInImage(Size imageSize, Size winStride) const;
	//依据img大小，窗口移动步伐，即窗口序号得到窗口在img中的位置
    Rect getWindow(Size imageSize, Size winStride, int idx) const;

	//buf为存储blockdata的内存空间，pt为block在parent img中的位置
    const float* getBlock(Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;

    vector<PixData> pixData;
    vector<BlockData> blockData;

	//以下的参数是为了充分利用重叠的block信息，避免重叠的block信息重复计算采用的一种缓存思想具体见后面代码 
    bool useCache;//是否存储已经计算的block信息
    vector<int> ymaxCached;//见后文
    Size winSize, cacheStride;//cacheStride认为等于blockStride,降低代码的复杂性
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;//img在扩展后图像中img原点关于扩展后原点偏置
    Mat_<float> blockCache;//待检测图像中以检测窗口进行横向扫描,所扫描的block信息存储在blockCache中 
    Mat_<uchar> blockCacheFlags;//判断当前block的信息blockCache中是否有存储,1：存储,于是直接调用；0：未存储,需要把信息存储到blockCache中 

    Mat grad, qangle;
    const HOGDescriptor* descriptor;
};

//默认的构造函数,不使用cache,块的直方图向量大小为0等
HOGCache::HOGCache()
{
    useCache = false;
    blockHistogramSize = count1 = count2 = count4 = 0;
    descriptor = 0;
}

//带参的初始化函数，采用内部的init函数进行初始化

HOGCache::HOGCache(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

//HOGCache结构体的初始化函数
//初始化主要包括：
//1、block中各像素对block四个bin的贡献权重，以及在存储空间中的位置 记录
//2、block的初始化，以及每个block在存储空间中的偏置及在检测窗口中的位置 记录
//3、其他参数的赋值
//并没有实际计算HOG
void HOGCache::init(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;

	/*--------------------------------------计算梯度----------------------------------------------*/  
    //返回值  
    //size：img.cols + paddingTL.width + paddingBR.width,img.rows + paddingTL.height + paddingBR.height,类型 CV_32FC2  
    //grad：梯度的模在与梯度方向相邻的两个bin的插值值  
    //qangle：与梯度方向相邻的两个bin的编号  
	
    //首先调用computeGradient()函数计算输入图像的权值梯度幅度图和角度量化图
    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
    //imgoffset是Point类型，而_paddingTL是Size类型，虽然类型不同，但是2者都是
    //一个二维坐标，所以是在opencv中是允许直接赋值的。
    imgoffset = _paddingTL;//16,24

    winSize = descriptor->winSize;//64*128
    Size blockSize = descriptor->blockSize;//16*16
    Size blockStride = descriptor->blockStride;//8*8
    Size cellSize = descriptor->cellSize;//8*8
    int i, j, nbins = descriptor->nbins;//9
    
	//rawBlockSize为block中包含像素点的个数
    int rawBlockSize = blockSize.width*blockSize.height;//16*16=256
    
    //nblocks为Size类型，其长和宽分别表示一个窗口中水平方向和垂直方向上block的
    //个数(需要考虑block在窗口中的移动)
	//这种算法非常直观，也许你会觉得可以和下面一样直接除，但是当(winSize.height - blockSize.height) % blockStride.height 不为0时，就不一定  
    //比如 blockSize=4,blockStride=3,winSize.width =9,那么直接除9/3=3，但是只能有两个block, 4|3|2,只能移动一次 
    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
                   (winSize.height - blockSize.height)/blockStride.height + 1);//7*15
    
	//ncells也是Size类型，其长和宽分别表示一个block中水平方向和垂直方向容纳下
    //的cell个数
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);//2*2
    //blockHistogramSize表示一个block中贡献给hog描述子向量的长度
    blockHistogramSize = ncells.width*ncells.height*nbins;//9*2*2

	
	//对于训练时,该段代码不起作用；对于检测时,该段代码可以提高运行速度。
	//在训练时,由于样本大小即等于检测窗口大小,因而不需要额外存储
	//但是在检测时由于待检测图像大于检测窗口,因而当检测窗口移动时,检测相邻检测窗口具有大量共同的block信息
	//为了节省时间,对于之前计算过大block信息,这里只需要调用,而对于未计算过的block信息,则重新计算并存储
	//其具体思路如下：假设待检测图像640*480,检测窗口为144*144			//这里就是先缩小检测区域，提高速度？？？？？？？？？
	//待检测图像水平方向有79个block,检测窗口垂直方向有17个block
	//于是由以下代码知道：blockCache为18*（79*36）=18*2844,blockCacheFlags为17*79,ymxcCached为17
	//以左上角代表检测窗口位置,当位于（0,0）时,第一次计算block信息,blockCache中是没有保存任何信息的。
	//当位于（0,0）时须计算（也以block左上角代表block位置）：
	//(0,0)---->(128,0) 信息均存储到blockCache中,分别为blockCache[0][0]--->blockCache[0][17*36],相应blockCacheFlags置1
	//(0,128)-->(128,128) blockCache[17][0]-->blockCache[17][17*36]
	//当检测窗口移动到（8,0）时,可以发现两个窗口中有大量信息是重复的,于是可以直接调用blockCache中相关block信息
	//并把（136,0）-->(136,128)新增列的block信息加到blockCache中,同时跟新blockCacheFlags
	//一直到窗口移到(624,0)进入到下一行(0,8),上述过程持续,于是blockCache中前17行存储了待检测图像中前17*79个block信息
	//当检测窗口移动到(624,0)时此时blockCache已经存储满了
	//当检测窗口移动到(0,8)时,第18行的信息怎么处理呢？
	//此时大家要留意的是第1行的block信息已经没有用啦,于是可以将第18行的信息替代第1行的信息。
	//当检测窗口不断横向扫描时,最新一行的信息总是会替代最旧一行的信息,如此反复,达到提高运行速度的目的
	//另外需要提到一点的是当block在pt=(x,y)=(0,0)-->(624,0)--->(0,128)---->(624.128)
	//可以用x/cacheStride=blockStride--->Canche_X,y/blockStride--->Cache_Y
	//从而从blockCache中取出对应的blockCache[Cache_Y][Cache_X*36]
	//当pt中y>128时,对应的第18行信息存储在第blockCache中的第0行
	//于是我们可以用取余的办法,y/blockStride%18--->Cache_Y,而Cache_X的计算不变
	//getblock函数中代码正是按该方法进行操作的 
    if( useCache )
    {
		//HOGCache的grad，qangle由discriptor->computerGradient得到
        //cacheStride= _cacheStride,即其大小是由参数传入的,表示的是窗口移动的大小
        //cacheSize长和宽表示扩充后的图像cache中，block在水平方向和垂直方向出现的个数
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
                       (winSize.height/cacheStride.height)+1);
        //blockCache为一个float型的Mat，注意其列数的值
        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        //blockCacheFlags为一个uchar型的Mat
        blockCacheFlags.create(cacheSize);
        size_t cacheRows = blockCache.rows;
        //ymaxCached为vector<int>类型
        //Mat::resize()为矩阵的一个方法，只是改变矩阵的行数，与单独的resize()函数不相同。
        ymaxCached.resize(cacheRows);
        //ymaxCached向量内部全部初始化为-1
        for(size_t ii = 0; ii < cacheRows; ii++ )
            ymaxCached[ii] = -1;
    }
    
    //weights为一个尺寸为blockSize的二维高斯表,下面的代码就是计算二维高斯的系数
	//sigma默认值为4
    Mat_<float> weights(blockSize);//16*16 高斯模板  
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

	//权重的二维高斯分布
    for(i = 0; i < blockSize.height; i++)
        for(j = 0; j < blockSize.width; j++)
        {
            float di = i - blockSize.height*0.5f;
            float dj = j - blockSize.width*0.5f;
            weights(i,j) = std::exp(-(di*di + dj*dj)*scale);
        }

    //vector<BlockData> blockData;而BlockData为HOGCache的一个结构体成员
    //nblocks.width*nblocks.height表示一个检测窗口中block的个数，
    //而cacheSize.width*cacheSize.heigh表示一个已经扩充的图片中的block的个数
    blockData.resize(nblocks.width*nblocks.height);//105个block
    //vector<PixData> pixData;同理，Pixdata也为HOGCache中的一个结构体成员
    //rawBlockSize表示每个block中像素点的个数
    //resize表示将其转换成列向量
    pixData.resize(rawBlockSize*3);//256*3(通道数)

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
    //count1,count2,count4分别表示block中同时对1个cell，2个cell，4个cell有贡献的像素点的个数。
    //作者用查找表的方法来计算。具体实现时是先执行HoGCache的初始化函数Init()
	//构造查找表，然后用getWindow()和getBlock()两个函数实现的表的查找

	count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )//16,先水平，再垂直  
        for( i = 0; i < blockSize.height; i++ )//16
        {
            PixData* data = 0;
            //cellX和cellY表示的是block内该像素点所在的cell横坐标和纵坐标索引，以小数的形式存在。
            //确定cell在block中的位置
			float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            //cvRound返回最接近参数的整数;cvFloor返回不大于参数的整数;cvCeil返回不小于参数的整数
            //icellX0和icellY0表示所在cell坐标索引，索引值为该像素点相邻cell的那个较小的cell索引
            //当然此处就是由整数的形式存在了。
            //按照默认的系数的话，icellX0和icellY0只可能取值-1,0,1,且当i和j<3.5时对应的值才取-1
            //当i和j>11.5时取值为1，其它时刻取值为0(注意i，j最大是15，从0开始的)
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            //此处的cellx和celly表示的是真实索引值与最近邻cell索引值之间的差，
            //为后面计算同一像素对不同cell中的hist权重的计算。
            cellX -= icellX0;
            cellY -= icellY0;
      
			//注意到unsigned，当icellX0=-1时，（unsigned）icellX0>2
			//(0~3,0~3)+(0~3,12~15)+(12~15,0~3)+(12~15,12~15)
			//(icellX0,icellY0,icellX1,icellY1)=(-1,-1,0,0),(-1,1,0,2),(1,-1,0,2),(1,1,2,2)===》条件4
			//(4~11,4~11)==》（0,0,1,1）==》条件1
			//(0~3,4~11)+(12~15,4~11)==》(-1,0,0,1)==》条件3
			//(4~11,0~3)+(4~11,12~15)==》(0,-1,1,0)==》条件2
			//情况2,3中元素对两个cell中的hist有贡献
			//(0~3,4~11):histofs=(0,9,0,0);(12~15,4~11):histofs=(18,27,0,0)
			//(4~11,0~3):histofs=(0,18,0,0);(4~11,12~15):hisofs=(9,27,0,0)
			//情况1中，元素对4个cell的hist有贡献,则会有4个hist及histofs,并且为(0,9,18,27)
			//情况4中，元素属于一个cell,则只有一个hist，对应的只有一个histofs:hist offset
			//分别应为：(0,0,0,0),(9,0,0,0),(18,0,0,0),(27,0,0,0)
			//对于权重的理解看后面的注释，选择第二种情况，其他可类推
            
			//满足这个if条件说明icellX0只能为0,也就是说block横坐标在(3.5,11.5)之间时
            //判断条件时特别小心，int 转成了 unsigned,(unsigned)(-1)=2^32-1，真对这作者无语  
			if( (unsigned)icellX0 < (unsigned)ncells.width &&
                (unsigned)icellX1 < (unsigned)ncells.width )
            {
               //满足这个if条件说明icellY0只能为0,也就是说block纵坐标在(3.5,11.5)之间时
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    //同时满足上面2个if语句的像素对4个cell都有权值贡献
                    //rawBlockSize表示的是1个block中存储像素点的个数
                    //而pixData的尺寸大小为block中像素点的3倍，其定义如下：pixData.resize(rawBlockSize*3);
                    //pixData的前面block像素大小的内存为存储只对block中一个cell有贡献的pixel；
					//中间block像素大小的内存存储对block中同时2个cell有贡献的pixel；
					//最后面的为对block中同时4个cell都有贡献的pixel
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    //下面计算出的结果为0
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;//cell 0 在整个block的bin中的偏移
                     //为该像素点对cell0的权重
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);//到对称中心的“距离”即cell 3 
                    //下面计算出的结果为18
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;//cell 1的偏移 2*9  
                    data->histWeights[1] = cellX*(1.f - cellY);//到对称中心的“距离”即 cell 2 
                    //下面计算出的结果为9
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;//cell 2的偏移 1*9  
                    data->histWeights[2] = (1.f - cellX)*cellY; //到对称中心的“距离”即 cell 1  
                    //下面计算出的结果为27
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;//cell 3的偏移3*9  
                    data->histWeights[3] = cellX*cellY;//到对称中心的“距离”即 cell 0  
                }
                else
                //满足这个else条件说明icellY0取-1或者1,也就是说block纵坐标在(0, 3.5)
                //和(11.5, 15)之间.
                //此时的像素点对相邻的2个cell有权重贡献
                {
                    data = &pixData[rawBlockSize + (count2++)];                    
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        //(unsigned)-1等于127>2，所以此处满足if条件时icellY0==1；
                        //icellY1==1;
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    //不满足if条件时，icellY0==-1;icellY1==0;
                    //当然了，这2种情况下icellX0==0;icellX1==1;
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            //当block中横坐标满足在(0, 3.5)和(11.5, 15)范围内时，即
            //icellX0==-1或==1
            else
            {
                
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    //icellX1=icllX0=1;
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }
                //当icllY0=0时，此时对2个cell有贡献
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
                //此时只对自身的cell有贡献
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
            //为什么每个block中i,j位置的gradOfs和qangleOfs都相同且是如下的计算公式呢？
            //那是因为输入的_img参数不是代表整幅图片而是检测窗口大小的图片，所以每个
            //检测窗口中关于block的信息可以看做是相同的
            data->gradOfs = (grad.cols*i + j)*2;//block窗口的（0,0）位置有相对于整个图像的偏移，此偏移为相对于block(0,0)的偏移 
            data->qangleOfs = (qangle.cols*i + j)*2;//计算方式很古怪，但是你画张图就明白了（grad.cols*i多算的==+j少算的），实际上 block窗口的（0,0）的offset加上此offset就可以直接在grad中找到对应的梯度  
            //每个block中i，j位置的权重都是固定的
            data->gradWeight = weights(i,j);//该点的高斯权值，大小与到block中心的距离成反比  
        }

    //保证所有的点都被扫描了一遍
    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData,碎片整理
    //数据合并  xxx.........yyy.........zzz.........->xxxyyyzzz..................  
    //（.表示未赋值空间，x为count1存储的数据，y为count2存储的数据...）
    //将pixData中按照内存排满，这样节省了2/3的内存
	//内存中存储顺序为：1,4,13,16/2,3,5,8,9,12,14,15/6,7,10,11区域像素的信息
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    //此时count2表示至多对2个cell有贡献的所有像素点的个数
    count2 += count1;
    //此时count4表示至多对4个cell有贡献的所有像素点的个数
    count4 += count2;

    //上面是初始化pixData,下面开始初始化blockData
    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            //histOfs表示该block对检测窗口贡献的hog描述变量起点在整个
            //变量中的坐标
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            //imgOffset表示该block的左上角在检测窗口中的坐标
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
        //一个检测窗口对应一个blockData内存，一个block对应一个pixData内存。
}


//buf:存储空间
//pt:block在parent img中的坐标，或偏置（左上角）
//只获取一个block中的信息：将256个像素的grad和angle信息变为36个bin的信息并保存
//pt为该block左上角在滑动窗口中的坐标，buf为指向检测窗口中blocData的指针
//函数返回一个block描述子的指针
const float* HOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
               (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

    if( useCache )	//默认未使用
    {
        //cacheStride可以认为和blockStride是一样的
        //保证所获取到HOGCache是我们所需要的，即在block移动过程中会出现
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        //cacheIdx表示的是block个数的坐标
        Point cacheIdx(pt.x/cacheStride.width,
                      (pt.y/cacheStride.height) % blockCache.rows);
        //ymaxCached的长度为一个检测窗口垂直方向上容纳的block个数
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            //取出blockCacheFlags的第cacheIdx.y行并且赋值为0
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        //blockHist指向该点对应block所贡献的hog描述子向量，初始值为空
        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;//64,128,256  
    //pt.x*2由于是2通道，记录block左上角对应在grad.data和qangle.data中的位置
    const float* gradPtr = (const float*)(grad.data + grad.step*pt.y) + pt.x*2;//block(0,0)在与其梯度方向相邻的两个bin上的插值分量  
    const uchar* qanglePtr = qangle.data + qangle.step*pt.y + pt.x*2;//与block(0,0)梯度方向相邻的两个bin的bin编号

    CV_Assert( blockHist != 0 );

	//blockHistogramSize=36
    for( k = 0; k < blockHistogramSize; k++ )
        blockHist[k] = 0.f;

	//遍历一个block中所有像素256个，以像素为单位取
	//一个像素包含：gradofs,qangleofs,gradweight,histofs[4],histweight[4]
	//pixData包含256个元素，blockData包含105个block
    const PixData* _pixData = &pixData[0];//pixData在init中已经计算好了，相对于block（0,0）的偏移

    //C1表示只对自己所在cell有贡献的点的个数
	//ADMP区域
    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        //a表示的是幅度指针
        const float* a = gradPtr + pk.gradOfs;//gradPtr起始地址，由不同输入Point pt而变化，pk.gradOfs偏置 
        float w = pk.gradWeight*pk.histWeights[0];
        //h表示的是相位指针
        const uchar* h = qanglePtr + pk.qangleOfs;

        //幅度有2个通道是因为每个像素点的幅值被分解到了其相邻的两个bin上了
        //相位有2个通道是因为每个像素点的相位的相邻处都有的2个bin的序号
        int h0 = h[0], h1 = h[1];//h[0]为angle所在bin的位置0~8，hist[h0]表示第h0个bin其中存储的是相应的幅度与权重
        float* hist = blockHist + pk.histOfs[0];//blockHist为buff的地址，histOfs即为偏置 
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        //hist中放的为加权的梯度值
        hist[h0] = t0; hist[h1] = t1;
    }
	//两个
	//BCEINPHL
    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        //因为此时的像素对2个cell有贡献，这是其中一个cell的贡献
        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        //另一个cell的贡献
        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    //四个
	//FGJK区域
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

//L2HysThreshold：先L2归一化，再限制所有的值的范围（0,0.2），再重新L2归一化
void HOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];

    size_t i, sz = blockHistogramSize;

    float sum = 0;

    //第一次归一化求的是平方和
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];

    //分母为平方和开根号+0.1
    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;

    for( i = 0, sum = 0; i < sz; i++ )
    {
        //第2次归一化是在第1次的基础上继续求平和和
        hist[i] = std::min(hist[i]*scale, thresh);//限制最大值为0.2
        sum += hist[i]*hist[i];
    }

	//在归一化一遍，使得各项平方和为1，即单位化
    scale = 1.f/(std::sqrt(sum)+1e-3f);

    //最终归一化结果
    for( i = 0; i < sz; i++ )
        hist[i] *= scale;

}


//返回测试图片中水平方向和垂直方向共有多少个检测窗口
Size HOGCache::windowsInImage(Size imageSize, Size winStride) const
{
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
                (imageSize.height - winSize.height)/winStride.height + 1);
}

//依据img大小，窗口移动步伐，即窗口序号得到窗口在img中的位置
//给定图片的大小，已经检测窗口滑动的大小和测试图片中的检测窗口的索引，得到该索引处
//检测窗口的尺寸，包括坐标信息
Rect HOGCache::getWindow(Size imageSize, Size winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;//商
    int x = idx - nwindowsX*y;//余数
    return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}

//img：待检测或计算的图像
//descriptors:Hog描述结构
//winStride:窗口移动步伐
//padding：扩充图像相关尺寸
//locations:对于正样本可以直接取(0,0),负样本为随机产生合理坐标范围内的点坐标
void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
                            Size winStride, Size padding,
                            const vector<Point>& locations) const
{
    //Size()表示长和宽都是0
	//若winStride.width=0,winStride.height=0，取(8,8)
    if( winStride == Size() )
        winStride = cellSize;
    //gcd为求最大公约数，如果采用默认值的话，则2者相同
	//alignSize(size_t sz, int n)
	//返回n的倍数中不小于sz的最小数，对padding.width进行修正
	//由默认参数有cacheStride=blockStride=(8,8)，padding.width=24,padding.height=16,所以也不需要修正，可忽视 
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));
	//正样本只有一个窗口，如果未扩充
	//负样本按论文中所说会随机产生10副图，若未扩充则会有10个窗口
    size_t nwindows = locations.size();
    //alignSize(m, n)返回n的倍数大于等于m的最小值
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	//当nwidows=0时扩充图像，之后再计算共有多少窗口area()=size.width*size.height,windowsInImage返回的是nwidth和nheight
	//在检测时会有用，由于检测时是不知道要计算哪块区域的，所以需要对整副图像需要多少窗口
	//训练时由于样本大小均为窗口大小,所以不需要额外存储block信息,则useCache=0,nwindows=1;
	//检测时由于待检测图像大于检测窗口大小,所以需要额外存储重复的block信息,则useCache=1,需要重新计算nwindows
	//detect函数中的useCache默认值为1,即检测时是需要额外存储block信息的
	//compute函数中的useCache默认值为0,detect会调用compute,会改变useCache的值
    if( !nwindows )
        //Mat::area()表示为Mat的面积
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();//一个hog的描述长度。一个窗口中特征向量大小：2*2*9*15*7=3780
    
	//resize()为改变矩阵的行数，如果减少矩阵的行数则只保留减少后的
    //那些行，如果是增加行数，则保留所有的行。
    //这里将描述子长度扩展到整幅图片
    descriptors.resize(dsize*nwindows);//注意到算法中样本大小为64*128，但实际上是有扩充的，实际特征向量还要乘上nwindows

	//descriptor存储分nwindows段，每段又分nblocks=105段，每段又有36个bi
    for( size_t i = 0; i < nwindows; i++ )
    {
        //descriptor为第i个检测窗口的描述子首位置。
        float* descriptor = &descriptors[i*dsize];
       
        Point pt0;
		//locations.empty()为空返回1
        //非空
        if( !locations.empty() )
        {
            pt0 = locations[i];
            //非法的点
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        //locations为空
        else
        {
            //pt0为没有扩充前图像对应的第i个检测窗口
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
			//imgOffset = Point(j*blockStride.width,i*blockStride.height),block在window中的位置
			//pt0:为img在parent img中的位置，注意到getBlock(pt,dst)中pt就是指的在parent img中的位置
            //pt为block的左上角相对检测图片的坐标
            Point pt = pt0 + bj.imgOffset;

			//histOfs=(j*nblocks.height + i)*blockHistogramSize,nblocks.height=15
            //dst为该block在整个测试图片的描述子的位置
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
    //hits里面存的是符合检测到目标的窗口的左上角顶点坐标
    hits.clear();
    if( svmDetector.empty() )
        return;

    if( winStride == Size() )//未指定winStride的情况下，winStride==（8,8）  
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));// gcd,求最大公约数，默认结果（8,8）  
    size_t nwindows = locations.size();// 默认：0  
	//对于我们自己设定的LTpading=BRpading=pading,进行调整使得pading的宽高与casheStride的宽高对齐，类似于4字节补充对齐 
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);//alignSize(m, n)，返回n的倍数中大于等于m的最小值  
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

	/*--------------------------------------------------------------------*/  
    //  1.计算梯度的模，方向  
    //  2.预先计算好了一个block的bin基偏移、高斯权重、插值距离  
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	//调用了，computeGradient，计算了pading后的梯度  
    //Note:尺度变化时，重新计算了梯度  
    //histOfs = (j*nblocks.height + i)*blockHistogramSize;  
    
    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();//整个img 的检测窗口数

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();//检测窗口内的block数
    int blockHistogramSize = cache.blockHistogramSize;//一个block的histogram的bin总数，2*2*9  
    size_t dsize = getDescriptorSize();//一个窗口的描述子总数  

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;// > 成立的情况，即svm的 惩罚项系数 C 不为0  
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
			//得到第i个检测窗口在pading之后的图像中的区域，这里减去pading,后面geitblock又pt += imgoffset; 
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        //svmVec指向svmDetector最前面那个元素
        const float* svmVec = &svmDetector[0];

        int j, k;

        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];//检测窗口中第j个block  
            // .histOfs = (j*nblocks.height + i)*blockHistogramSize;  
            // .imgOffset = Point(j*blockStride.width,i*blockStride.height);  
			Point pt = pt0 + bj.imgOffset;//得到第i个检测窗口中第j个block在pading之后的图像中的TL坐标
            
            //vec为测试图片pt处的block贡献的描述子指针
            const float* vec = cache.getBlock(pt, &blockHist[0]);

			//计算到分类超平面的距离
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

//不用保留检测到目标的可信度，即权重
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
        //将原图片进行缩放
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        vector<Point> locations;
        vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            //smallerImg只是构造一个指针，并没有复制数据
            Mat smallerImg(sz, img.type(), smallerImgBuf.data);
            //没有尺寸缩放
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            //有尺寸缩放
            else
                resize(img, smallerImg, sz);
			//dst的内存空间超过src时，dst的空间是不是并没有缩小呢，  
            //也就是说是不是先释放内存，再按照新的size重新申请,从程序上看一直霸占原始内存空间才能起到减少内存申请释放所耗费的时间  
            
			//该函数实际上是将返回的值存在locations和histWeights中
            //其中locations存的是目标区域的左上角坐标
            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            for( size_t j = 0; j < locations.size(); j++ )
            {
                //保存目标区域
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                //保存缩放尺寸
                if (scales) {
                    scales->push_back(scale);
                }
            }
            //保存svm计算后的结果值
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
	//要使检测窗口的尺度变大有两种方案，法一：图像尺寸不变，增大检测窗口的大小；法二：反过来，检测窗口不变，缩小图片  
    //这里使用的正是第二种方法  
	
	//计算出最大层数，基本是将图像缩小，即认为样本尺度已经很小了，实际的行人只会大于样本尺寸，小于样本尺寸的行人无法检测 
    //nlevels默认的是64层
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(img.cols/scale) < winSize.width ||	//小于64层尺度的尺度数由是由图形的尺寸和scale0决定的，
            cvRound(img.rows/scale) < winSize.height || //当图像缩放到已经小于检测窗口时就已经不能在增加尺度了  
            scale0 <= 1 )
            break;
        //只考虑测试图片尺寸比检测窗口尺寸大的情况
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
    
    //TBB并行计算	http://blog.csdn.net/zoufeiyy/article/details/1887579  
    parallel_for(BlockedRange(0, (int)levelScale.size()),
                 HOGInvoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &tempWeights, &tempScales));
    
	//将tempScales中的内容复制到foundScales中；back_inserter是指在指定参数迭代器的末尾插入数据
    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    //容器的clear()方法是指移除容器中所有的数据
    foundLocations.clear();
    //将候选目标窗口保存在foundLocations中
    std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    foundWeights.clear();
    //将候选目标可信度保存在foundWeights中
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));

    if ( useMeanshiftGrouping )
    {
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
    }
    else
    {
        //对矩形框进行聚类
        groupRectangles(foundLocations, (int)finalThreshold, 0.2);
    }
}

//不考虑目标的置信度
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
       //返回detector数组的从头到尾构成的向量
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
        //返回detector的首尾构成的向量
        return vector<float>(detector, detector + sizeof(detector)/sizeof(detector[0]));
}
}