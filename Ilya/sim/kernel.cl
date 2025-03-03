#define gp(A, X) A[X.x*world_sizes[1] + X.y]  
#define sp(A, X) A[X.x*world_sizes[1] + X.y]  

#define diff 0.05f


__kernel void movement_update(__global float* Posd, __global float* movement, __global const int* world_sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = pos.x*world_sizes[1] + pos.y;
    float POS = Posd[ind];

    int2 left = (pos + (int2) (1, 0));
    left.x = max(min(left.x, world_sizes[0] - 1), 0);
    int2 up = (pos + (int2) (0, 1));
    up.y = max(min(up.y, world_sizes[1] - 1), 0);
    int2 mleft = (pos - (int2) (1, 0));
    mleft.x = max(min(mleft.x, world_sizes[0] - 1), 0);   
    int2 mup = (pos - (int2) (0, 1));
    mup.y = max(min(mup.y , world_sizes[1] - 1), 0);
    
    float ddx = 2.0f*((gp(Posd, mleft) + gp(Posd, left))/2.0f - POS)/(diff*diff);

    float ddy = 2.0f*((gp(Posd, mup) + gp(Posd, up))/2.0f - POS)/(diff*diff);

    float F = 10.0f*(ddx + ddy);

    //float dmov = movement[ind] + F*diff*diff * max(sin((float)pos.x*pos.x / 10000.0f) * sin((float)pos.y / 10.0f) + 0.1, 0.0)/2.0f;
    float dmov = movement[ind] + F*diff*diff * max(length(((float2) (pos.x, pos.y))/ (float2)(world_sizes[0], world_sizes[0]) - (float2)(0.5f, 0.25)) - 0.02f, 0.0005f);
    movement[ind] = dmov;
}

__kernel void Posd_update(__global float* Posd, __global float* next_Posd, __global float* movement, __global const int* world_sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = pos.x*world_sizes[1] + pos.y;
    float POS = Posd[ind];
    next_Posd[ind] = POS + movement[ind]*diff;
}


/*
__kernel void upd_buffers(__global float* buff1, __global float* buff2){
    int i = get_global_id(0);
    float super_a = buff1[i];
    float super_b = buff2[i];
    buff1[i] = super_b;
    buff2[i] = super_a;

}*/

__kernel void get_image(__global float* psi, __global float* movement, __global int* image, __global const int* world_sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = pos.x*world_sizes[1] + pos.y;
    float r = psi[ind];
    float m = movement[ind];
    int C = max((int)(255.0f*min(r + 50.0f, 100.0f)/100.0f), 0);
    int C1 = max((int)(255.0f*min(m + 50.0f, 100.0f)/100.0f), 0);
    image[ind*3 + 0] = C;
    image[ind*3 + 1] = C;
    image[ind*3 + 2] = C;
}


__kernel void clear(__global float* Posd, __global float* next_Posd, __global float* movement, __global const int* world_sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = pos.x*world_sizes[1] + pos.y;
    next_Posd[ind] = 0.0f;
    Posd[ind] = 0.f;
    movement[ind] = 0.f;
}

__kernel void draw_wawe( __global float* Posd, __global float* movement, __global const int* world_sizes, __global const int* position, __global const float* radius, __global float* value){
    int2 pos = (int2)(get_global_id(0) + position[0], get_global_id(1) + position[1]);
    pos.x = min(max(pos.x, 0), world_sizes[0]);
    pos.y = min(max(pos.y, 0), world_sizes[1]);
    float2 p = (float2)(get_global_id(0), get_global_id(1)) - *radius;
    int ind = pos.x*world_sizes[1] + pos.y;
    float L = length(p);
    if (L <= radius[0]*1.0f){
        Posd[ind] += value[0]*(1.0f - L/radius[0]);
        movement[ind] += value[1]*(1.0f - L/radius[0]);
    }
}

__kernel void draw_particle( __global float* Posd, __global float* movement, __global const int* world_sizes, __global const int* position, __global const float* radius, __global float* value){
    int2 pos = (int2)(get_global_id(0) + position[0], get_global_id(1) + position[1]);
    pos.x = min(max(pos.x, 0), world_sizes[0]);
    pos.y = min(max(pos.y, 0), world_sizes[1]);
    float2 p = (float2)(get_global_id(0), get_global_id(1)) - *radius;
    int ind = pos.x*world_sizes[1] + pos.y;
    float L = length(p);
    if (L <= radius[0]){
        Posd[ind] += value[0]*(sin(10.0*(p.x)))*(1.0f - L/radius[0]);
        //Posd[ind] += value[0]*sin(L/radius[0]);

    }
}

/*
__kernel void draw_U(__global float* psi, __global int* world_sizes, __global float* value){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = pos.x*world_sizes[1] + pos.y;
    psi[ind*2 + 0] = value[0];
    psi[ind*2 + 1] = value[1];
}*/