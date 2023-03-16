# Inelastic Neutron Scattering Benchmark

Machine learning (ML) and quantum mechanics have been combined by a team from ISIS Neutron 
and Muon Source and SciML to develop a new method that can analyse neutron scattering 
experiments and understand the magnetic structure of materials. Inelastic neutron scattering (INS) is a powerful technique that allows scientists to probe the atomic level structure of solids, it also reveals important information about how magnetic spins are arranged across individual atomic sites in the system. This level of understanding is critical for developing a range of exotic new technologies, such as quantum computers and high temperature superconductors.
Typically an INS experiment is analysed by careful, meticulous examination of the experimental 
results by a trained physicist. The process involves taking many samples from the experimental data and comparing these samples to simulations based on quantum mechanics. This procedure is both computationally and intellectually highly demanding and analysis of a single dataset can often take years to complete properly. 

Now the team at SciML have developed a ML approach that can learn from simulated data and make 
predictions on the experimental data in a fraction of the time. It was also important for the scientists at ISIS that the methods are able to make statements about why the ML model gives the result that it does, so explainable ML was used to analyse the data. The results were very exciting, with the ML model identifying the same small regions of the INS data and correctly predicting the same magnetic structure of the materials tested as the scientists from ISIS had published some years before. While the analysis of the original data took more than two years, the entire process for training and running the ML model takes less than a week.

The success of the ML model was highly dependent on access to SCARF resources. First to develop 
the training dataset, quantum mechanics simulations were run in parallel on the SCARF nodes. 
Testing and developing of the ML models was then split between SCARF and PEARL machines 
both using GPUs to efficiently run the neural networks. 

Publications
KT Butler, MD Le, J Thiyagalingam, TG Perring Journal of Physics: Condensed Matter 33 (19), 
194006, 2021

<!--
Machine learning (ML) and quantum mechanics have been combined by a team from ISIS Neutron 
and Muon Source and SciML to develop a new method that can analyse neutron scattering 
experiments and understand the magnetic structure of materials. Inelastic neutron scattering (INS) is a powerful technique that allows scientists to probe the atomic level structure of solids, it also reveals important information about how magnetic spins are arranged across individual atomic sites in the system. This level of understanding is critical for developing a range of exotic new technologies, such as quantum computers and high temperature superconductors.
Typically an INS experiment is analysed by careful, meticulous examination of the experimental 
results by a trained physicist. The process involves taking many samples from the experimental data and comparing these samples to simulations based on quantum mechanics. This procedure is both computationally and intellectually highly demanding and analysis of a single dataset can often take years to complete properly. 

Now the team at SciML have developed a ML approach that can learn from simulated data and make 
predictions on the experimental data in a fraction of the time. It was also important for the scientists at ISIS that the methods are able to make statements about why the ML model gives the result that it does, so explainable ML was used to analyse the data. The results were very exciting, with the ML model identifying the same small regions of the INS data and correctly predicting the same magnetic structure of the materials tested as the scientists from ISIS had published some years before. While the analysis of the original data took more than two years, the entire process for training and running the ML model takes less than a week.

The success of the ML model was highly dependent on access to SCARF resources. First to develop 
the training dataset, quantum mechanics simulations were run in parallel on the SCARF nodes. 
Testing and developing of the ML models was then split between SCARF and PEARL machines 
both using GPUs to efficiently run the neural networks. 

Publications
KT Butler, MD Le, J Thiyagalingam, TG Perring Journal of Physics: Condensed Matter 33 (19), 
194006, 2021

-->