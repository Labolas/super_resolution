#include <iostream>

#include <visp/vpDebug.h>
#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpDisplayX.h>

using namespace std ;

static void 
RGBtoYUV(const vpImage<vpRGBa> &I,
	 vpImage<unsigned char> &Y, vpImage<unsigned char> &Cb, vpImage<unsigned char> &Cr)
{
  int h=I.getHeight(), w=I.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      Y[i][j]  =   0.2125 * I[i][j].R + 0.7154 * I[i][j].G + 0.0721 * I[i][j].B;
      Cb[i][j] = - 0.115  * I[i][j].R - 0.385  * I[i][j].G + 0.5    * I[i][j].B + 128;
      Cr[i][j] =   0.5    * I[i][j].R - 0.454  * I[i][j].G - 0.046  * I[i][j].B + 128;
    }
}

static void
createDico(const vpImage<unsigned char> &comp)
{
  int h=comp.getHeight(), w=comp.getWidth();

  // dans une dizaine d'images, passage VGG16
  // récupérations de cartes 
  
  // 
  
}

int main()
{
  int h=319, w=480;
  vpImage<vpRGBa> I(h,w,0);
  
  vpImage<unsigned char> Y (h,w);
  vpImage<unsigned char> Cb(h,w);
  vpImage<unsigned char> Cr(h,w);

  vpImageIo::read(I,"../img/lion.jpg") ;
  
  // convertion to YUV
  RGBtoYUV(I, Y, Cb, Cr);
  
  vpDisplayX d1(I) ;
  vpDisplayX d2(Y) ;
  vpDisplayX d3(Cb) ;
  vpDisplayX d4(Cr) ;
  vpDisplay::display(I) ;
  vpDisplay::display(Y) ;
  vpDisplay::display(Cb) ;
  vpDisplay::display(Cr) ;
  vpDisplay::flush(I) ;
  vpDisplay::flush(Y) ;
  vpDisplay::flush(Cb) ;
  vpDisplay::flush(Cr) ;
  vpDisplay::getClick(I) ;


  return 0;
}
