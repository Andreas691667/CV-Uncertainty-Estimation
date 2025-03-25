class feature_specification():
    def __init__(self,feature_id, feature_type_id, left, top, right, bottom):
        self.id      = feature_id
        self.type_id = feature_type_id
        self.left = left
        self.top  = top
        self.right = right
        self.bottom = bottom

        if feature_type_id == 0:
            # This is a facial feature
            self.beta = 1

        elif feature_type_id == 1:
            # This is a palm feature
            self.beta = 1

        elif feature_type_id == 2:
            # This is a legs feature
            self.beta = 1
        
        self.assigned_person_id = None
        self.found_overlapping_prediction_flag = False
