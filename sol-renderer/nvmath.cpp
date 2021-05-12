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

#include "nvmath.h"

float4x4 Quat2Mat(quat& q)
{
    float xx, xy, xz, xw, yy, yz, yw, zz, zw, ww;

    xx = q.x * q.x;
    xy = q.x * q.y;
    xz = q.x * q.z;
    xw = q.x * q.w;
    yy = q.y * q.y;
    yz = q.y * q.z;
    yw = q.y * q.w;
    zz = q.z * q.z;
    zw = q.z * q.w;
    ww = q.w * q.w;

    return make_float4x4(ww + xx - yy - zz, 2.0f*(xy + zw), 2.0f*(xz - yw), 0.0f, 
                         2.0f*(xy - zw), ww - xx + yy - zz, 2.0f*(yz + xw), 0.0f,
                         2.0f*(xz + yw), 2.0f*(yz - xw), ww - xx - yy + zz, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f);
}


quat QuatMul(quat& Q1, quat& Q2)
{
    quat Qr;

    Qr.w = Q1.w * Q2.w - Q1.x * Q2.x - Q1.y * Q2.y - Q1.z * Q2.z;
    Qr.x = Q1.x * Q2.w + Q1.w * Q2.x - Q1.z * Q2.y + Q1.y * Q2.z;
    Qr.y = Q1.y * Q2.w + Q1.z * Q2.x + Q1.w * Q2.y - Q1.x * Q2.z;
    Qr.z = Q1.z * Q2.w - Q1.y * Q2.x + Q1.x * Q2.y + Q1.w * Q2.z;

    return Qr;
}


float4x4 lookAt
(
    const float3& eye,
    const float3& at,
    const float3& up
)
{
    float3 f = normalize(at - eye);
    float3 s = normalize(cross(f, up));
    float3 u = cross(s, f);
    
    return make_float4x4(s.x, u.x, -f.x, 0.0f,
                s.y, u.y, -f.y, 0.0f,
                s.z, u.z, -f.z, 0.0f,
                -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f);
}


float4x4 lookAtInv
(
    const float3& eye,
    const float3& at,
    const float3& up
)
{
    float3 f = normalize(at - eye);
    float3 s = normalize(cross(f, up));
    float3 u = cross(s, f);
    
    return make_float4x4(s.x, s.y, s.z, 0.0f,
                u.x, u.y, u.z, 0.0f,
                -f.x, -f.y, -f.z, 0.0f,
                eye.x, eye.y, eye.z, 1.0f);
}



float4x4 perspective
(
    const float& fovy,
    const float& aspect,
    const float& zNear,
    const float& zFar
)
{
    float const rad = 0.01745329f * fovy;
    float tanHalfFovy = tanf(0.5f * rad);

    return make_float4x4(1.0f / (aspect * tanHalfFovy), 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f / (tanHalfFovy), 0.0f, 0.0f,
                0.0f, 0.0f, -(zFar + zNear) / (zFar - zNear), -1.0f,
                0.0f, 0.0f, -(2.0f * zFar * zNear) / (zFar - zNear), 0.0f);

}


float4x4 perspectiveInv
(
    const float& fovy,
    const float& aspect,
    const float& zNear,
    const float& zFar
)
{
    float const rad = 0.01745329f * fovy;
    float tanHalfFovy = tanf(0.5f * rad);

    return make_float4x4(aspect * tanHalfFovy, 0.0f, 0.0f, 0.0f,
                0.0f, tanHalfFovy, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, -(zFar + zNear) / (2.0f * zFar * zNear),
                0.0f, 0.0f, 1.0f, (zFar * zNear) / (2.0f * zFar * zNear));
}
