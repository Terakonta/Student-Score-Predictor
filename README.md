# Student-Score-Predictor
This project used diagnostic questions to predict student performance (Part of university assignment)


Online education services, such as Khan Academy and Coursera, provide a broader audience with
access to high-quality education. On these platforms, students can learn new materials by watching
a lecture, reading course material, and talking to instructors in a forum. However, one disadvantage
of the online platform is that it is challenging to measure students' understanding of the course
material. To deal with this issue, many online education platforms include an assessment component
to ensure that students understand the core topics. The assessment component is often composed
of diagnostic questions, each a multiple choice question with one correct answer. The diagnostic
question is designed so that each of the incorrect answers highlights a common misconception. When students incorrectly answer
the diagnostic question, it reveals the nature of their misconception and, by understanding these
misconceptions, the platform can offer additional guidance to help resolve them. In this project, we
will build machine learning algorithms to predict whether a student can correctly
answer a specific diagnostic question based on the student's previous answers to other questions
and other students' responses. Predicting the correctness of students' answers to as yet unseen
diagnostic questions helps estimate the student's ability level in a personalized education platform.
Moreover, these predictions form the groundwork for many advanced customized tasks. For in-
stance, using the predicted correctness, the online platform can automatically recommend a set of
diagnostic questions of appropriate difficulty that fit the student's background and learning status.  

Here, I will use kNN, NN and one parameter IRT to see which one performs better. Prediction accuracy will be 
measure by doing:  
(# of correct predictions / # of total predictions)
