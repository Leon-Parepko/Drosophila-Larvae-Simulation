#define get_activity(t, i) activity[*t**activity_size + i]
#define W(i, j) w[i*activity_size[0] + j]

__kernel void difference_matrix(__global float* a, __global float *b, __global int *size_a, __global int *size_b, __global float* result){
    int x = get_global_id(0);
    int y = get_global_id(1);

    result[x**size_b + y] = a[x] - b[y];
}

float pushing_function(float w){
    return 1.0f/(1.0f + w*w);
}

float pulling_function(float m){
    return 1.0f - 1.0f/(1.0f + m*m);
}

__kernel void simulate_clasters(__global const float *activity, __global const float *w, __global float *positions, __global float *next_positions, __global const int* activity_size, __global const int *t, __global const float* _dt){
    int ind = get_global_id(0);
    const float dt = *_dt;

    float2 direction = (float2) (0.f, 0.f);
    float2 speed = (float2) (0.f, 0.f);

    for (int i = 0; i < *activity_size; i++){
        if (i != ind){
            direction.x = positions[ind + 0] - positions[i + 0];
            direction.y = positions[ind + 1] - positions[i + 1];
            float r = length(direction);
            direction = normalize(direction);

            speed += direction*(pushing_function(W(ind, i))/r - pulling_function(W(ind, i) * (get_activity(t, i) - get_activity(t - 1, i))));
        }
    }   
    next_positions[ind + 0] = positions[ind + 0] + speed.x*dt;
    next_positions[ind + 1] = positions[ind + 1] + speed.y*dt;
}