# From buying to building AI workstations

![enter image description here](https://github.com/svishnu88/blogs/blob/master/bram-van-oost-4xM5cytsdMo-unsplash.jpg?raw=true)

I started my deep learning journey back in 2014. For a few months, I spent a crazy 0.9$ for a GPU machine on AWS to run model prototyping. For many of us, this is often a huge cost particularly when it goes from our own pockets. By 2015 it was clear that renting machines on the cloud for primary deep learning model prototyping is not a financially viable option. I bought my first GPU powered workstation in mid of 2016.

From 2015 to 2019, I have worked on several AI-related projects in the fields of Computer vision and Natural language processing. When working with several teams, I started observing a pattern of challenges that most data scientists and teams face. Some of the key challenges are

- Identifying the right tools for the problem. Often teams expect DL/AI to solve very complex problems.

- Once it is determined that the problem can be solved using DL, the next big challenge is getting the right data.

- Prototyping models on either On-Premise or Cloud.

- Test the model

- Deploy the model.

  

A lot of these problems are domain-specific and difficult to generalize. The problem of choosing the On-Premise AI cluster vs Cloud cluster is interesting. I often tell my clients that the time needed for finishing the project is directly proportional to the number of GPUs the team has access to. So it becomes tricky to come up with the right size for the On-Premise cluster. Relying on cloud platforms like AWS, GCP, and Azure is often very expensive for model prototyping as it involves a lot of experimentation with different architectures and hyperparameter tuning. Running AI workloads on the cloud has been made considerably easier by a lot of tools like SageMaker in recent years. When it comes to On-premise not many tools have been built to make lives of AI teams easier.

Some of the key challenges that organizations could face when building and managing their AI Cluster.

  

- Picking the right hardware.

- Installing and managing the right software stack.

  

## Picking the right hardware

Here we have 2 choices, buying a pre-built machine or building one ourselves.

  

### Buying

We have a few choices when we want to buy a pre-built machine. Buy an Nvidia DGX workstation built by the same company that powers most of AI applications today. Often it is not affordable for many of us. 

![enter image description here](https://github.com/svishnu88/blogs/blob/master/nvidia-dgx.png?raw=true)

There are some pre-built options available in India, but they are mostly backed by traditional hardware sales company. A lot of these machines are built without an understanding of how the machines are going to be used. Some of the hardware choices would limit using more graphic cards or using libraries that accelerate several of tasks including preprocessing. Outside India, particularly in the US, you have a few good choices who sell well designed pre-built machines for deep learning. Some of them do not sell in India and some of them ship to India, but the products would attract a crazy shipping charge and customs making it 30 % more expensive.

  

### Building

  

Building your own machines would be a suitable choice provided teams have enough time to do research and understand the different requirements behind building a workstation. Often times it requires teams with multiple specialties like data scientists (Understanding their use cases), hardware engineers (Choice of graphic cards, a brand of the processor) and electrical team (Gpu's are very power-hungry) to decide on the components to buy and build. 

![enter image description here](https://github.com/svishnu88/blogs/blob/master/danist-8Gg2Ne_uTcM-unsplash.jpg?raw=true)

The kind of problem we are trying to solve will have a direct impact on the choice of graphic cards. For computer vision problems involving large image, resolutions may require GPU's with large memory. If we are more focused on problems that can be solved by RNN, we would need GPUs with more memory as parallelizing the algorithms is quite challenging.

  

## Software stacks

  

With software stacks choosing the right combination of different software required and managing them over a long period of time comes with its own complexity.

  

## Installing software stack

Teams would love to try different frameworks like fastai, PyTorch, TensorFlow which comes in different versions. Each framework, some times even the same framework of different versions comes with a lot of dependencies making life difficult for different teams managing AI clusters. Below are some tweets from top deep learning practitioners talking about their pain setting up the core software.

![enter image description here](https://github.com/svishnu88/blogs/blob/master/problem1&2.png?raw=true)


Though there are several blogs, articles, and documents that exist showing how to set up different software, they rarely work because of different parts(OS, Framework versions, GPU model) involved.

  

## Managing software stack

  

Your AI cluster needs to be often shared among teams with different requirements. Some teams would need TensorFlow 1 version and others may need TensorFlow version 2. Each comes with its own dependency making it difficult to install and manage. In my experience working across projects for multiple clients, I have observed that 30 to 40 percent of time gets wasted in making the multiple frameworks work for different teams. Another challenge is most of the frameworks may not release the GPU memory immediately after use. So teams could be starving of GPU's even when they are not used, causing extended project timelines.

  

# Birth of jarvislabs.ai

I left my last job in the mid of 2019 and continued with helping a few clients as an Independent AI consultant. After spending a lot of money on different cloud platforms while participating in different Kaggle competitions, I strongly believed, it was time to add a more powerful workstation. So I started looking for a pre-built workstation that suits my needs and I realized I may not be able to buy without paying a hefty premium(Customs, International shipping charges). So after talking to a couple of my friends (some of them turned to be co-founders and customers), we decided to design a workstation that is optimal for the majority of deep learning workloads. As we discussed earlier how building a workstation is half the battle, and the other battle is installing and managing software stack which is very painful.

  

Managing deep learning stack on cloud is comparatively easier due to the nature of cloud platforms. We wanted to bring these benefits to data science teams. We wanted our workstations to come with software that ensures the installation and management of different frameworks as easy as clicking a button. We should be able to create virtual instances with frameworks of our choice, focus on building models and never worry about Cuda issues.

# Meet Jarvis Manager
![enter image description here](https://raw.githubusercontent.com/svishnu88/blogs/master/JarvisManager.png)


Every Jarvis Workstation comes with our Jarvis Manager pre-installed. It helps you create a virtual instance in your AI Custer in just a few clicks. Each virtual instance is optimized so that you can squeeze the maximum efficiency from each GPU. We wanted to build something that we use every day, and Jarvis Manager is helping do the same.

With every Jarvis Workstation we want you to save all the t
Want to know more about us, write to us hello@jarvislabs.ai
