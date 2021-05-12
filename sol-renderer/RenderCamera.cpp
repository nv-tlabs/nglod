/******************************************************************************
 * The MIT License (MIT)

 * Copyright (c) 2021, NVIDIA CORPORATION.

 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#include "RenderCamera.h"


RenderCamera::RenderCamera(void)
{
    float4x4 I = make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f);
    SetWorldMatrix(I);
    m_bLMouseDown = false;
    m_t = tan(0.125f * 180.0f);
}


RenderCamera::~RenderCamera(void)
{
}

void RenderCamera::OnResize(uint width, uint height)
{
    m_Width = width;
    m_Height = height;
    m_q0 = make_float4(0.0f, 0.0f, 0.0f, 1.0f) ;
}


float3 RenderCamera::mouse2vector(int mx, int my)
{
    float xsize = 0.5f*m_Width;
    float ysize = 0.5f*m_Height;

    float3 v = make_float3(xsize - (float)mx, (float)my - ysize, 0.0f);

    if (xsize >= ysize) 
    {
        v.x = 1.1f*v.x/ysize;
        v.y = 1.1f*v.y/ysize;
    }
    else 
    {
        v.x = 1.1f*v.x/xsize;
        v.y = 1.1f*v.y/xsize;
    }

    float tmp = v.x*v.x + v.y*v.y;

    if (tmp < 1.0f) 
    {
        v.z = (float)(-sqrt(1.0 - tmp));
    }
    else 
    {
        tmp = (float)sqrt(tmp);
        v.x = v.x/tmp;
        v.y = v.y/tmp;
        v.z = 0.0f;
    }

    return v;
}


void RenderCamera::OnMouseDown(int x, int y)
{
    m_bLMouseDown = true;
    m_v0 = mouse2vector(x, y);
}


void RenderCamera::OnMouseUp()
{
    if (m_bLMouseDown)
    {
        m_bLMouseDown = false;
        m_q0 = m_q;
    }
}


void RenderCamera::OnMouseMove(int x, int y)
{
    if (m_bLMouseDown)
    {
        if (m_v0.x == 0.0f && m_v0.y == 0.0f && m_v0.z == 0.0f) // special case bug fix
        {
            m_v0 = mouse2vector(x, y);
        }

        m_v1 = mouse2vector(x, y);

        /* v0 cross m_v1 */
        m_q1.x = m_v1.y*m_v0.z - m_v0.y*m_v1.z;
        m_q1.y = m_v1.z*m_v0.x - m_v0.z*m_v1.x;
        m_q1.z = m_v1.x*m_v0.y - m_v0.x*m_v1.y;

        /* v0 dot m_v1 */
        m_q1.w = -(m_v0.x*m_v1.x + m_v0.y*m_v1.y + m_v0.z*m_v1.z);

        m_q = QuatMul(m_q1, m_q0);

        float4x4 Q = Quat2Mat(m_q);

        float3 t = m_At - m_Eye;
        float len = length(t);

        float4x4 T = make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, len, 1.0f);

        float4x4 Tinv = make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, -len, 1.0f);

        float4x4 W = m_mView * T * Q  * Tinv * m_mViewInv;

        SetWorldMatrix(W);
    }
}


void RenderCamera::OnMouseWheel(short zDelta)
{
    if(zDelta < 0)
    {
        m_t -= 0.05;
    }
    else
    {
        m_t += 0.05f;
    }

    m_t = fminf(m_t, 8.01);
    m_t = fmaxf(m_t, -0.5);

    float FOV = (float)((2*atan(m_t >= 0.0 ? 1.0/pow(1+m_t,2):pow(m_t-1,2))));

    SetFOV(57.29577951f * FOV);
}


void RenderCamera::SetWorldMatrix( float4x4& world )
{
    m_mWorld = world;
    m_mWorldInv = transpose(m_mWorld);
}


void RenderCamera::SetViewMatrix( float3& pvEyePt, float3& pvLookatPt, float3& pvUpVec )
{
    m_mView = lookAt( pvEyePt, pvLookatPt, pvUpVec );
    m_mViewInv = lookAtInv( pvEyePt, pvLookatPt, pvUpVec );

    m_Eye = pvEyePt;
    m_At = pvLookatPt;
    m_Up = pvUpVec;
}


void RenderCamera::SetProjMatrix( float fFOV, float fAspect, float fNearPlane, float fFarPlane )
{
    // Set attributes for the projection matrix
    m_fFOV = fFOV;
    m_fAspect = fAspect;
    m_fNearPlane = fNearPlane;
    m_fFarPlane = fFarPlane;

    m_mProj = perspective( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
    m_mProjInv = perspectiveInv( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
}


void RenderCamera::SetFOV( float fFOV )
{
    // Set attributes for the projection matrix
    m_fFOV = fFOV;

    m_mProj = perspective( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
    m_mProjInv = perspectiveInv( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
}

void RenderCamera::SetNearFar( float fNearPlane, float fFarPlane )
{
    m_fNearPlane = fNearPlane;
    m_fFarPlane = fFarPlane;

    m_mProj = perspective( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
    m_mProjInv = perspectiveInv( m_fFOV, m_fAspect, m_fNearPlane, m_fFarPlane );
}
