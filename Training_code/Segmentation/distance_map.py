import numpy as np

def compute_distance(first_point,second_point,len_area):

    points_list = [[] for i in range(len_area)]
    points_list[len_area-1] += [first_point,second_point]

    if second_point[1]!=first_point[1]:

        coeff = float((second_point[0]-first_point[0]))/float((second_point[1]-first_point[1]))

        if(first_point[1] <= second_point[1]):
            for i in range(1,abs(second_point[1]-first_point[1])):
                list = [first_point[0]+ int(i*coeff), first_point[1] + i]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

        else:
            for i in range(1,abs(first_point[1]-second_point[1])):
                list = [first_point[0]- int(i*coeff), first_point[1] - i]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

    else:

        if first_point[0] <= second_point[0]:
            for i in range(1,abs(first_point[0]-second_point[0])):
                list=[first_point[0]+i,first_point[1]]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

        else:
            for i in range(1,abs(first_point[0]-second_point[0])):
                list=[first_point[0]-i,first_point[1]]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

    if second_point[0]!=first_point[0]:

        coeff2 = float((second_point[1]-first_point[1]))/float((second_point[0]-first_point[0]))

        if first_point[0] <= second_point[0]:
            for i in range(1,abs(second_point[0]-first_point[0])):
                list = [first_point[0]+i ,first_point[1]+int(i*coeff2)]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

        else:
            for i in range(1,abs(second_point[0]-first_point[0])):
                list = [first_point[0]-i ,first_point[1]-int(i*coeff2)]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

    else:

        if first_point[1] <= second_point[1]:
            for i in range(1,abs(first_point[0]-second_point[0])):
                list=[first_point[0],first_point[1]+i]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

        else:
            for i in range(1,abs(first_point[0]-second_point[0])):
                list=[first_point[0],first_point[1]-i]
                points_list[len_area-1].append(list)
                for i in range(2,len_area+1):
                    points_list[len_area-i] += appendPointsAround(list,i-1)

    return points_list

def appendPointsAround(central_point,size):
    list_points = []
    for i in range(-size, size+1):
        point_1 = [central_point[0]-size , central_point[1] + i]
        point_2 = [central_point[0]+size , central_point[1] + i]
        point_3 = [central_point[0] , central_point[1]-size]
        point_4 = [central_point[0] , central_point[1]+size]
        list_points.append(point_1)
        list_points.append(point_2)
        list_points.append(point_3)
        list_points.append(point_4)
    return list_points
