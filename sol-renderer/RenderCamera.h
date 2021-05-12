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

#pragma once
#include "nvmath.h"

class RenderCamera
{
public:
    RenderCamera(void);
    ~RenderCamera(void);

    void OnResize(uint width, uint height);
    void OnMouseDown(int x, int y);
    void OnMouseUp();
    void OnMouseMove(int x, int y);
    void OnMouseWheel(short zDelta);

    void SetWorldMatrix(float4x4& world);
    void SetViewMatrix(float3& pvEyePt, float3& pvLookatPt, float3& pvUpVec );
    void SetProjMatrix( float fFOV, float fAspect, float fNearPlane, float fFarPlane );
    void SetFOV( float fFOV );
    void SetNearFar( float fNearPlane, float fFarPlane );

    float4x4 GetWorldMatrix() const
    {
        return m_mWorld;
    }

    float4x4 GetViewMatrix() const
    {
        return m_mView;
    }

    float4x4 GetProjMatrix() const
    {
        return m_mProj;
    }

    float4x4 GetWorldInvMatrix() const
    {
        return m_mWorldInv;
    }

    float4x4 GetViewInvMatrix() const
    {
        return m_mViewInv;
    }

    float4x4 GetProjInvMatrix() const
    {
        return m_mProjInv;
    }

    float GetFOV() const
    {
        return m_fFOV;
    }

    float4x4 m_mWorld;              // Projection matrix
    float4x4 m_mView;              // View matrix 
    float4x4 m_mProj;              // Projection matrix

    float4x4 m_mWorldInv;
    float4x4 m_mViewInv;    
    float4x4 m_mProjInv;    


private:
    uint m_Width;
    uint m_Height;


    float3 m_v0;
    float3 m_v1;


    quat m_q;
    quat m_q0;
    quat m_q1;

    double  m_t;

    bool m_bLMouseDown;

    float3 m_Eye;
    float3 m_At;
    float3 m_Up;

    float m_fFOV ;
    float m_fAspect;
    float m_fNearPlane;
    float m_fFarPlane;


    float3 mouse2vector(int mx, int my);
};

