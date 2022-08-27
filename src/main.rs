use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{
    device_type_text, get_all_devices, vendor_id_text, Device, CL_DEVICE_TYPE_GPU,
};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_SVM_ATOMICS, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_int, CL_BLOCKING};
use opencl3::Result;
use rand::Rng;
use std::io::{self};
use std::ptr;
use std::time::SystemTime;

const COMPARE_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
kernel void compare_kernel(global const int* source, global int* counts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(source[i]>source[j])
    {
        atomic_inc(&counts[i]);
    }
}
"#;

const COMPARE_KERNEL_NAME: &str = "compare_kernel";

const ASSIGN_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
kernel void assign_kernel(global const int* source, global const int* counts, global int* results)
{
    int my_id = get_global_id(0);
    int i = 0;
    int expected;
    do
    {
        expected = -1;
        expected = atomic_cmpxchg(&results[counts[my_id]+i],expected,source[my_id]);
        i++;
    }while(expected != -1 && (counts[my_id]+i < get_global_size(0)));
}
"#;

const ASSIGN_KERNEL_NAME: &str = "assign_kernel";

fn main() -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    //print device info
    println!("\tCL_DEVICE_VENDOR: {}", device.vendor()?);
    let vendor_id = device.vendor_id()?;
    println!(
        "\tCL_DEVICE_VENDOR_ID: {:X}, {}",
        vendor_id,
        vendor_id_text(vendor_id)
    );
    println!("\tCL_DEVICE_NAME: {}", device.name()?);
    println!("\tCL_DEVICE_VERSION: {}", device.version()?);
    let device_type = device.dev_type()?;
    println!(
        "\tCL_DEVICE_TYPE: {:X}, {}",
        device_type,
        device_type_text(device_type)
    );
    println!("\tCL_DEVICE_PROFILE: {}", device.profile()?);
    println!("\tCL_DEVICE_EXTENSIONS: {}", device.extensions()?);
    println!(
        "\tCL_DEVICE_OPENCL_C_VERSION: {:?}",
        device.opencl_c_version()?
    );
    println!(
        "\tCL_DEVICE_BUILT_IN_KERNELS: {}",
        device.built_in_kernels()?
    );
    println!(
        "\tCL_DEVICE_SVM_CAPABILITIES: {:X}",
        device.svm_mem_capability()
    );
    println!();

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
    )
    .expect("CommandQueue::create failed");

    //Build Kernels
    let compare_program =
        Program::create_and_build_from_source(&context, COMPARE_KERNEL, "-cl-std=CL1.2")
            .expect("Compare Kernel Failed to Build");
    let compare_kernel = Kernel::create(&compare_program, COMPARE_KERNEL_NAME)
        .expect("Compare Kernel Create Failed");

    let assign_program =
        Program::create_and_build_from_source(&context, ASSIGN_KERNEL, "-cl-std=CL1.2")
            .expect("Assign Kernel Failed to Build");
    let assign_kernel =
        Kernel::create(&assign_program, ASSIGN_KERNEL_NAME).expect("Assign Kernel Create Failed");

    println!("Choose an max value:");
    let mut max_sort_value = String::new();
    io::stdin()
        .read_line(&mut max_sort_value)
        .expect("Failed to read line");
    let max_sort_value: i32 = match max_sort_value.trim().parse() {
        Ok(num) => num,
        Err(_) => 0,
    };

    println!("Choose a array size:");
    let mut max_array_size = String::new();
    io::stdin()
        .read_line(&mut max_array_size)
        .expect("Failed to read line");
    let max_array_size: i32 = match max_array_size.trim().parse() {
        Ok(num) => num,
        Err(_) => 0,
    };

    let max_array_size = max_array_size.try_into().unwrap();

    //Input Data
    let mut input = vec![0; max_array_size];
    for i in 0..max_array_size {
        input[i] = rand::thread_rng().gen_range(0..max_sort_value);
    }

    let mut input_buffer =
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, max_array_size, ptr::null_mut())
            .expect("Failed to make Input Buffer");
    let mut count_buffer = Buffer::<cl_int>::create(&context, 0, max_array_size, ptr::null_mut())
        .expect("Failed to make Count Buffer");
    let mut result_buffer =
        Buffer::<cl_int>::create(&context, CL_MEM_WRITE_ONLY, max_array_size, ptr::null_mut())
            .expect("Failed to make Result Buffer");

    println!("Line: {}", line!());

    let input_write_event = queue
        .enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, &input, &[])
        .expect("Failed to queue Input Buffer");
    let count_write_event = queue
        .enqueue_write_buffer(
            &mut count_buffer,
            CL_BLOCKING,
            0,
            &vec![0; max_array_size],
            &[],
        )
        .expect("Failed to queue Count Buffer");
    let result_write_event = queue
        .enqueue_write_buffer(
            &mut result_buffer,
            CL_BLOCKING,
            0,
            &vec![-1; max_array_size],
            &[],
        )
        .expect("Failed to queue Write Buffer");

    let compare_kernel_event = ExecuteKernel::new(&compare_kernel)
        .set_arg(&input_buffer)
        .set_arg(&count_buffer)
        .set_global_work_sizes(&[max_array_size, max_array_size])
        .set_wait_event(&input_write_event)
        .set_wait_event(&count_write_event)
        .enqueue_nd_range(&queue)?;

    let mut events: Vec<cl_event> = Vec::default();
    events.push(compare_kernel_event.get());

    let assign_kernel_event = ExecuteKernel::new(&assign_kernel)
        .set_arg(&input_buffer)
        .set_arg(&count_buffer)
        .set_arg(&result_buffer)
        .set_global_work_size(max_array_size)
        .set_wait_event(&result_write_event)
        .set_wait_event(&compare_kernel_event)
        .enqueue_nd_range(&queue)?;

    events.push(assign_kernel_event.get());

    let mut result = vec![0; max_array_size];
    let read_event =
        queue.enqueue_read_buffer(&result_buffer, CL_BLOCKING, 0, &mut result, &events)?;

    read_event.wait()?;

    println!("Array to Sort");
    for i in 0..max_array_size {
        print!("{} ", input[i]);
    }
    print!("\n\n");
    println!("Parallel Sorted Array");
    for i in 0..max_array_size {
        print!("{} ", result[i]);
    }
    print!("\n");

    let compare_start_time = compare_kernel_event.profiling_command_start()?;
    let compare_end_time = compare_kernel_event.profiling_command_end()?;
    let assign_start_time = compare_kernel_event.profiling_command_start()?;
    let assign_end_time = compare_kernel_event.profiling_command_end()?;
    let duration = (compare_end_time - compare_start_time) + (assign_end_time - assign_start_time);
    println!("Parallel execution duration (ns): {}\n", duration);

    let mut merge_sort_input = vec![0; 0];
    for i in 0..max_array_size {
        merge_sort_input.push(input[i]);
    }
    let start = SystemTime::now();
    let merge_sort_result = merge_sort(&merge_sort_input);
    let end = SystemTime::now();
    let elapsed = end.duration_since(start);

    println!("MergeSort Sorted Array");
    for i in 0..max_array_size {
        print!("{} ", merge_sort_result[i]);
    }
    print!("\n");
    println!(
        "MergeSort execution duration (ns): {}\n",
        elapsed.unwrap_or_default().as_nanos()
    );

    //Everything is fine
    Ok(())
}

fn merge_sort(vec: &Vec<i32>) -> Vec<i32> {
    if vec.len() < 2 {
        vec.to_vec()
    } else {
        let size = vec.len() / 2;
        let left = merge_sort(&vec[0..size].to_vec());
        let right = merge_sort(&vec[size..].to_vec());
        let merged = merge(&left, &right);

        merged
    }
}

fn merge(left: &Vec<i32>, right: &Vec<i32>) -> Vec<i32> {
    let mut i = 0;
    let mut j = 0;
    let mut merged: Vec<i32> = Vec::new();

    while i < left.len() && j < right.len() {
        if left[i] < right[j] {
            merged.push(left[i]);
            i = i + 1;
        } else {
            merged.push(right[j]);
            j = j + 1;
        }
    }

    if i < left.len() {
        while i < left.len() {
            merged.push(left[i]);
            i = i + 1;
        }
    }

    if j < right.len() {
        while j < right.len() {
            merged.push(right[j]);
            j = j + 1;
        }
    }

    merged
}
