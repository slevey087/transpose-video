from manim import *
import numpy as np
from math import sqrt


def ComputeSVD(matrix, flip_if_negative_determinant=True):
    u, s, vt = np.linalg.svd(matrix)

    # Turn s from vector of singular values to matrix
    s = np.diag(s)

    # reflect u and v if they both have reflections
    # Takes negative of first column of u and corresponding first row of v
    if flip_if_negative_determinant and np.linalg.det(u) < 0 and np.linalg.det(vt) < 0:
        u[:, 0] *= -1
        vt[0, :] *= -1

    return u, s, vt


def MatrixFromSVD(R1, S, R2):
    """
        Computes a matrix using SVD parameters.
        R1 and R2 are rotation angles (radians)
        S is a 2 parameter vector, which are the singular values.
    """
    r1 = np.array([
        [np.cos(R1), np.sin(R1)],
        [-np.sin(R1), np.cos(R1)]
    ])

    s = np.diag(S)

    r2 = np.array([
        [np.cos(R2), -np.sin(R2)],
        [np.sin(R2), np.cos(R2)]
    ])

    matrix = np.matmul(r2, np.matmul(s, r1))
    return matrix


# Primary example
pematrix = MatrixFromSVD(-PI/7, [1.7, 0.6], PI/5)
# pex = [-1.4,-1]
pex = [-1, 1.5]
# pev = [0,1.7]
pev = [1.75, -0.3]


def GetBaseline(item):
    orig_text = item.get_tex_string()
    orig_center = item.get_center()
    orig_bottom = item.get_bottom()

    temp_text = MathTex(orig_text,"+")
    temp_text.scale((orig_center[1]-orig_bottom[1])/(temp_text[0].get_center()[1]-temp_text[0].get_bottom()[1]))
    temp_text.shift(orig_center-temp_text[0].get_center())
        
    baseline = temp_text[1].get_bottom()[1]
    return baseline


def AlignBaseline(item, reference):
    BaselineDifference = GetBaseline(reference) - GetBaseline(item) 
    item.shift([0,BaselineDifference,0])
    return item


def FixMatrixBaselines(matrix,anchors):
    for row, anchor in zip(matrix.get_rows(), anchors):
        anchor_base = GetBaseline(row[anchor])
        for item in row:
            base = GetBaseline(item)
            item.shift([0,anchor_base-base,0])



def GetBaseGapDifference(item1, item2):
    BaseGap1 = item1.get_center()[1]-GetBaseline(item1)
    BaseGap2 = item2.get_center()[1]-GetBaseline(item2)
    return BaseGap1 - BaseGap2


def TransformBuilder(mobject1, mobject2, formula, default_transform=ReplacementTransform, default_creator=Write, default_remover=FadeOut):
    """
        Returns transformations which convert mobject1 into mobject2, according to formula.

        formula should be a list of tuples, one for each transformation.
        A tuple consists of four elements: a locator for the initial mobject portion, a locator for the final mobject portion, and a code for what kind of transform to do, and any optional kwargs for the transform as a dict
        To write a new element, put Null for the source.
        To remove an existing element, put Null for the destination.
    """
    def recurse(l, locator):
        if len(locator) == 1: return l[locator[0]]
        else: return recurse(l[locator[0]], locator[1:])

    anims = []
    for item in formula:
        source_locator      = item[0]
        destination_locator = item[1]
        transform           = item[2] if len(item) >= 3 else None
        kwargs              = item[3] if len(item) == 4 else {}
        
        # Figure out source
        if source_locator is None: source = None
        elif not isinstance(source_locator, list): source = mobject1[source_locator]
        else: source = recurse(mobject1, source_locator)

        # Figure out destination
        if destination_locator is None: destination = None
        elif not isinstance(destination_locator, list): destination = mobject2[destination_locator]
        else: destination = recurse(mobject2, destination_locator)

        
        # Default transforms
        if source is None:
            transform = transform or default_creator
            anims.append(transform(destination, **kwargs))
        
        elif destination is None:
            transform = transform or default_remover
            anims.append(transform(source, **kwargs))
        
        else:
            transform = transform or default_transform
            anims.append(transform(source, destination, **kwargs))
    
    return anims



def matrix_product(matrix1, matrix2, h_buff):
    """
        Returns a matrix product. Doesn't actualy multiiply numbers, just writes entries next to each other.
    """
    rows1 = len(matrix1.get_rows())
    cols1 = len(matrix1.get_rows()[0])
    rows2 = len(matrix2.get_rows())
    cols2 = len(matrix2.get_rows()[0])
    if cols1 != rows2:
        raise ValueError(
            "The number of columns in matrix1 must be equal to the number of rows in matrix2.")
    entries1 = matrix1.get_entries()
    entries2 = matrix2.get_entries()
    entries3 = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            entry = ""
            for k in range(cols1):
                entry += entries1[i*cols1+k].get_tex_string() + \
                    entries2[k*cols2+j].get_tex_string()+"+"
            row.append(entry[:-1])
        entries3.append(row)
    return Matrix(entries3, h_buff=h_buff)


def multiply_matrix(scene, matrix1, matrix2, matrix3=None, h_buff=2.15, location=None):
    """
        Animates the product of two matrices, already on the screen.
        Returns (matrix3, equals_sign)
    """
    equals_sign = MathTex("=").next_to(
        matrix2 if location is None else location, RIGHT)
    matrix3 = matrix_product(
        matrix1, matrix2, h_buff=h_buff) if matrix3 is None else matrix3
    matrix3 = matrix3.next_to(equals_sign, RIGHT)

    scene.play(Write(equals_sign), Write(matrix3.get_brackets()))

    for i in range(2):
        for j in range(2):
            row = matrix1.get_rows()[i]
            col = matrix2.get_columns()[j]
            source = VGroup(row, col)
            scene.play(Indicate(row), Indicate(col), TransformFromCopy(
                source, matrix3.get_rows()[i][j]))

    return (matrix3, equals_sign)



def transpose21(scene, matrix, with_diagonal=False, diagonal_flash=False, fade_when_done=False):
    # Create transpose
    matrixt = Matrix([[entry.get_tex_string()
                     for entry in matrix.get_entries()]])
    matrixt.move_to(matrix)

    # Turn blue
    elements = matrix.get_entries()
    elementst = matrixt.get_entries()
    off_elements = [elements[i] for i in [0, 1]]
    scene.play(*[element.animate.set_color(BLUE)
               for element in off_elements], run_time=0.5)

    # Prep element animations
    tmap = {"origin": [0, 1], "destination": [1, 0]}
    entry_animations = [elements[origin].animate.move_to(elementst[destination].get_center(
    )) for (origin, destination) in zip(tmap["origin"], tmap["destination"])]

    # Prep bracket animations
    for (bracket, brackett) in zip(matrix.get_brackets(), matrixt.get_brackets()):
        bracket.target = brackett
    bracket_animations = [MoveToTarget(bracket)
                          for bracket in matrix.get_brackets()]

    # Animate
    scene.play(*(entry_animations + bracket_animations), run_time=1.3)

    # Turn white
    scene.play(*[element.animate.set_color(WHITE)
               for element in off_elements], run_time=0.5)

    if fade_when_done:
        scene.play(FadeOut(matrix))
    return matrix


def transpose2(scene, matrices, with_diagonal=False, diagonal_flash=False, fade_when_done=False):
    # Color blue, transpose, color white
    if not isinstance(matrices, list):
        matrices = [matrices]

    if with_diagonal:
        diagonals = [DashedLine(matrix.get_entries()[0].get_center(), matrix.get_entries(
        )[3].get_center(), dash_length=0.25, dashed_ratio=0.6) for matrix in matrices]
        scene.play(*[Create(diagonal) for diagonal in diagonals])

        if diagonal_flash:
            scene.remove(*diagonals)
            scene.play(*[Create(diagonal) for diagonal in diagonals])

    element_sets = [matrix.get_entries() for matrix in matrices]

    blue_anims = [item for sublist in [[elements[1].animate.set_color(
        BLUE), elements[2].animate.set_color(BLUE)] for elements in element_sets] for item in sublist]
    scene.play(*blue_anims, run_time=0.5)

    move_anims = [item for sublist in [[
        elements[1].animate.move_to(elements[2].get_center()+[0,GetBaseGapDifference(elements[1],elements[2]),0]), 
        elements[2].animate.move_to(elements[1].get_center()+[0,GetBaseGapDifference(elements[2],elements[1]),0])] for elements in element_sets
    ] for item in sublist]
    scene.play(*move_anims, run_time=1)

    white_anims = [item for sublist in [[elements[1].animate.set_color(
        WHITE), elements[2].animate.set_color(WHITE)] for elements in element_sets] for item in sublist]
    scene.play(*white_anims, run_time=0.6)

    fades = []
    if with_diagonal:
        fades += [FadeOut(diagonal) for diagonal in diagonals]
    if fade_when_done:
        fades += [FadeOut(matrix) for matrix in matrices]
    if len(fades):
        scene.play(AnimationGroup(*fades))
    return matrices


def transpose3(scene, matrix, with_diagonal=False, diagonal_flash=False, fade_when_done=False):
    # Color blue, transpose, color white
    if with_diagonal:
        diagonal = DashedLine(matrix.get_entries()[0].get_center(), matrix.get_entries()[
                              8].get_center(), dash_length=0.25, dashed_ratio=0.6)
        scene.play(Create(diagonal))

        if diagonal_flash:
            scene.remove(diagonal)
            scene.play(Create(diagonal))

    # Turn blue
    elements = matrix.get_entries()
    off_elements = [elements[i] for i in [1, 2, 3, 5, 6, 7]]
    scene.play(*[element.animate.set_color(BLUE)
               for element in off_elements], run_time=0.5)

    # Move
    tmap = {"origin": [1, 2, 3, 5, 6, 7], "destination": [3, 6, 1, 7, 2, 5]}
    entry_animations = [elements[origin].animate.move_to(elements[destination].get_center(
    )) for (origin, destination) in zip(tmap["origin"], tmap["destination"])]
    scene.play(*entry_animations, run_time=1.3)

    # Turn white
    scene.play(*[element.animate.set_color(WHITE)
               for element in off_elements], run_time=0.5)

    fades = []
    if with_diagonal:
        fades += [FadeOut(diagonal)]
    if fade_when_done:
        fades += [FadeOut(matrix)]
    if len(fades):
        scene.play(AnimationGroup(*fades))
    return matrix


def transpose32(scene, matrix, matrixt, with_diagonal=False, diagonal_flash=False, fade_when_done=False):
    # Align if not already:
    matrixt.align_to(matrix, UP+LEFT)

    # Color blue, transpose, color white
    if with_diagonal:
        diagonal = DashedLine(matrix.get_entries()[0].get_center(), matrix.get_entries()[
                              3].get_center(), dash_length=0.25, dashed_ratio=0.6)
        scene.play(Create(diagonal))

        if diagonal_flash:
            scene.remove(diagonal)
            scene.play(Create(diagonal))

    # Turn blue
    elements = matrix.get_entries()
    elementst = matrixt.get_entries()
    off_elements = [elements[i] for i in [1, 2, 4, 5]]
    scene.play(*[element.animate.set_color(BLUE)
               for element in off_elements], run_time=0.5)

    # Prep element animations
    tmap = {"origin": [1, 2, 4, 5], "destination": [3, 1, 2, 5]}
    entry_animations = [elements[origin].animate.move_to(elementst[destination].get_center(
    )) for (origin, destination) in zip(tmap["origin"], tmap["destination"])]

    # Prep bracket animations
    for (bracket, brackett) in zip(matrix.get_brackets(), matrixt.get_brackets()):
        bracket.target = brackett
    bracket_animations = [MoveToTarget(bracket)
                          for bracket in matrix.get_brackets()]

    # Animate
    scene.play(*(entry_animations + bracket_animations), run_time=1.3)

    # Turn white
    scene.play(*[element.animate.set_color(WHITE)
               for element in off_elements], run_time=0.5)

    fades = []
    if with_diagonal:
        fades += [FadeOut(diagonal)]
    if fade_when_done:
        fades += [FadeOut(matrix)]
    if len(fades):
        scene.play(AnimationGroup(*fades))
    return matrix


def property_animation(scene, Tex1, Tex2):
    """ returns (formula, underline)"""
    formula1 = MathTex(Tex1, "", font_size=100)
    formula2 = MathTex(Tex1, Tex2, font_size=100)

    # Write and animate formula
    scene.play(Write(formula1))
    scene.wait(1.5)
    scene.play(ReplacementTransform(formula1, formula2))
    scene.wait(2)

    # Prep to move and underline formula
    formula2t = formula2.copy()
    formula2t.set_font_size(60)
    formula2t.shift(3*UP)
    underline = Line(start=formula2t.get_left()+DOWN*0.5+LEFT*0.25,
                     end=formula2t.get_right()+DOWN*0.5+RIGHT*0.25, color=BLUE)

    # Move and underline formula
    scene.play(ReplacementTransform(formula2, formula2t), Write(underline))

    return (formula2t, underline)


def SVDAnim(scene, matrix, circle=None, draw_scaling_vectors=True, invert_s=False, invert_r=False, added_anims=[[], [], []]):
    u, s, vt = ComputeSVD(matrix)

    if invert_s:
        s = np.linalg.inv(s)
    if invert_r:
        u, vt = np.linalg.inv(vt), np.linalg.inv(u)
        

    # Add circle
    if circle is None:
        circle = Circle(radius=1).set_color(BLUE)
        scene.play(Create(circle))
        scene.add_transformable_mobject(circle)
        scene.wait()

    # Apply SVD transformation 1
    scene.moving_mobjects = []
    scene.apply_matrix(vt, added_anims=added_anims[0])

    # Add basis vectors
    if draw_scaling_vectors:
        i1, j1 = scene.get_basis_vectors(
            i_hat_color=BLUE, j_hat_color=BLUE).set_opacity(0.6)
        scene.play(GrowArrow(i1), GrowArrow(j1))
        scene.moving_vectors += [i1, j1]

    # SVD 2, remove basis
    scene.moving_mobjects = []
    scene.apply_matrix(s, added_anims=added_anims[1])
    if draw_scaling_vectors:
        scene.play(FadeOut(VGroup(i1, j1)))
        scene.moving_vectors.remove(i1)
        scene.moving_vectors.remove(j1)

    # SVD 3
    scene.moving_mobjects = []
    scene.apply_matrix(u, added_anims=added_anims[2])

    return circle




def SVDAnimByAxis(scene, axis, matrix, circle=None, draw_scaling_vectors=True, basis_vectors=[], invert_s=False, invert_r=False, added_anims=[[], [], []]):
    u, s, vt = ComputeSVD(matrix)

    if invert_s:
        s = np.linalg.inv(s)
    if invert_r:
        u, vt = np.linalg.inv(vt), np.linalg.inv(u)

    # Add circle
    if circle is None:
        circle = Circle(radius=axis.get_x_unit_size()).set_color(BLUE).move_to(axis.c2p(0,0))
        scene.play(Create(circle))        
        scene.wait()

    # Apply SVD transformation 1        
    func = lambda point: axis.c2p(*(vt @ np.array(axis.p2c(point)).T))    
    anims = [
        ApplyPointwiseFunction(func, axis.background_lines),
        ApplyPointwiseFunction(func, circle),
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors
    ] + added_anims[0]
    scene.play(*anims)
    

    # # Add basis vectors
    if draw_scaling_vectors:
        i1 = Arrow(ORIGIN, [1,0,0], color=BLUE)
        i1.put_start_and_end_on(axis.get_origin(), axis.c2p(1,0))
        j1 = Arrow(ORIGIN, [1,0,0], color=BLUE)        
        j1.put_start_and_end_on(axis.get_origin(), axis.c2p(0,1))
        scene.add(i1, j1)                  
        scene.play(GrowArrow(i1), GrowArrow(j1))        
            

    # # SVD 2, remove basis    
    func = lambda point: axis.c2p(*(s @ np.array(axis.p2c(point)).T))    
    anims = [
        ApplyPointwiseFunction(func, axis.background_lines),
        ApplyPointwiseFunction(func, circle),
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors        
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in [i1, j1]
    ] + added_anims[1]
    scene.play(*anims)
    if draw_scaling_vectors: scene.play(FadeOut(VGroup(i1, j1)))

    # # SVD 3
    func = lambda point: axis.c2p(*(u @ np.array(axis.p2c(point)).T))    
    anims = [
        ApplyPointwiseFunction(func, axis.background_lines),
        ApplyPointwiseFunction(func, circle),
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors        
    ] + added_anims[2]
    scene.play(*anims)    

    return circle



def SVDAnimByAxisByVT(scene, axis, matrix, circle=None, draw_scaling_vectors=True, basis_vectors=[], invert_s=False, invert_r=False, added_anims=[[], [], []], updater=None):
    u, s, vt = ComputeSVD(matrix)

    if invert_s:
        s = np.linalg.inv(s)
    if invert_r:
        u, vt = np.linalg.inv(vt), np.linalg.inv(u)

    # Add circle
    if circle is None:
        circle = Circle(radius=axis.get_x_unit_size()).set_color(BLUE).move_to(axis.c2p(0,0))
        scene.play(Create(circle))        
        scene.wait()

    # Apply SVD transformation 1        
    func = lambda point: axis.c2p(*(vt @ np.array(axis.p2c(point)).T))    
    v1 = ValueTracker(0)
    for line in axis.background_lines:
        line.clear_updaters()            
        line.original_points = line.get_start_and_end()            
        line.add_updater(lambda l: l.put_start_and_end_on(l.original_points[0],l.original_points[1]).apply_matrix(np.array([[1+v1.get_value()*(vt[0][0]-1),vt[0][1]*v1.get_value()], [vt[1][0]*v1.get_value(),1+v1.get_value()*(vt[1][1]-1)]]), about_point=axis.get_center()))
        line.add_updater(updater)    
    anims = [
        v1.animate.set_value(1),
        ApplyPointwiseFunction(func, circle)
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors
    ] + added_anims[0]
    scene.play(*anims, run_time=3)
    

    # # Add basis vectors
    if draw_scaling_vectors:
        i1 = Arrow(ORIGIN, [1,0,0], color=BLUE)
        i1.put_start_and_end_on(axis.get_origin(), axis.c2p(1,0))
        j1 = Arrow(ORIGIN, [1,0,0], color=BLUE)        
        j1.put_start_and_end_on(axis.get_origin(), axis.c2p(0,1))
        scene.add(i1, j1)                  
        scene.play(GrowArrow(i1), GrowArrow(j1))        
            

    # # SVD 2, remove basis    
    func = lambda point: axis.c2p(*(s @ np.array(axis.p2c(point)).T))    
    v2 = ValueTracker(0)
    for line in axis.background_lines:
        line.clear_updaters()            
        line.original_points = line.get_start_and_end()            
        line.add_updater(lambda l: l.put_start_and_end_on(l.original_points[0],l.original_points[1]).apply_matrix(np.array([[1+v2.get_value()*(s[0][0]-1),s[0][1]*v2.get_value()], [s[1][0]*v2.get_value(),1+v2.get_value()*(s[1][1]-1)]]), about_point=axis.get_center()))
        line.add_updater(updater)        
    anims = [
        v2.animate.set_value(1),
        ApplyPointwiseFunction(func, circle),
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors        
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in [i1, j1]
    ] + added_anims[1]
    scene.play(*anims, run_time=3)
    if draw_scaling_vectors: scene.play(FadeOut(VGroup(i1, j1)))

    # # SVD 3
    func = lambda point: axis.c2p(*(u @ np.array(axis.p2c(point)).T))    
    v3 = ValueTracker(0)
    for line in axis.background_lines:
        line.clear_updaters()            
        line.original_points = line.get_start_and_end()            
        line.add_updater(lambda l: l.put_start_and_end_on(l.original_points[0],l.original_points[1]).apply_matrix(np.array([[1+v3.get_value()*(u[0][0]-1),u[0][1]*v3.get_value()], [u[1][0]*v3.get_value(),1+v3.get_value()*(u[1][1]-1)]]), about_point=axis.get_center()))
        line.add_updater(updater)                
    anims = [
        v3.animate.set_value(1),
        ApplyPointwiseFunction(func, circle),
    ] + [
        Transform(v, Arrow(ORIGIN, [1,1,0], color=v.get_color()).put_start_and_end_on(axis.get_origin(), func(v.get_end())),run_time=3) for v in basis_vectors        
    ] + added_anims[1]
    scene.play(*anims, run_time=3)    

    return circle



def SVD2Transpose(scene, matrix, x, v, matrix_run_time=3, creation_run_time=1, draw_scaling_vectors=True, circle1=None, added_anims=[[], [], []]):
    """
        Animates the SVD of a dot product.
        Returns a VGroup of the circles it draws,
        so you can decide how to tear them down.
    """
    matrix = np.array(matrix)
    u, s, vt = ComputeSVD(matrix)
    sinv = np.linalg.inv(s)

    # add circles
    if circle1 is None:
        circle1 = Circle(radius=1).set_color(
            BLUE) if circle1 is None else circle1
        scene.play(Create(circle1), run_time=creation_run_time)

    circle2 = Circle(radius=1).set_color(BLUE)
    scene.add(circle2)
    scene.add_transformable_mobject(circle1)
    scene.add_transformable_mobject(circle2)

    # First transformation in SVD (same for both)
    scene.moving_mobjects = []
    scene.apply_matrix(vt, run_time=matrix_run_time,
                       added_anims=added_anims[0])

    # Add basis vectors, 2 sets
    if draw_scaling_vectors:
        i1, j1 = scene.get_basis_vectors(
            i_hat_color=BLUE, j_hat_color=BLUE).set_opacity(0.6)
        i2, j2 = scene.get_basis_vectors(
            i_hat_color=BLUE, j_hat_color=BLUE_A).set_opacity(0.6)
        scene.play(GrowArrow(i1), GrowArrow(i2), run_time=creation_run_time)
        scene.play(GrowArrow(j1), GrowArrow(j2), run_time=creation_run_time)
        scene.moving_vectors += [i1, j1, i2, j2]

    # SVD 2, get forward and backward s anims separately then run simultaneously
    scene.moving_vectors.remove(v)
    if draw_scaling_vectors:
        scene.moving_vectors.remove(i2)
        scene.moving_vectors.remove(j2)
    scene.transformable_mobjects.remove(circle2)

    s_func = scene.get_matrix_transformation(s)
    vecs1_anim = scene.get_vector_movement(s_func)
    circle1_anim = ApplyPointwiseFunction(s_func, circle1)

    scene.moving_vectors.remove(x)
    if draw_scaling_vectors:
        scene.moving_vectors.remove(i1)
        scene.moving_vectors.remove(j1)
    scene.moving_vectors.append(v)
    if draw_scaling_vectors:
        scene.moving_vectors.append(i2)
        scene.moving_vectors.append(j2)
    scene.transformable_mobjects.remove(circle1)
    scene.transformable_mobjects.append(circle2)

    sinv_func = scene.get_matrix_transformation(sinv)
    vecs2_anim = scene.get_vector_movement(sinv_func)
    circle2_anim = ApplyPointwiseFunction(sinv_func, circle2)

    scene.play(
        circle1_anim,
        circle2_anim,
        vecs1_anim,
        vecs2_anim,
        *added_anims[1], run_time=matrix_run_time)
    if draw_scaling_vectors:
        scene.play(FadeOut(VGroup(i1, j1, i2, j2)))
        for vec in [i2, j2]:
            scene.moving_vectors.remove(vec)
    scene.add_transformable_mobject(circle1)
    scene.moving_vectors.append(x)

    # SVD 3
    scene.moving_mobjects = []
    scene.apply_matrix(u, run_time=matrix_run_time, added_anims=added_anims[2])

    return VGroup(circle1, circle2)



def find_intersections(axes, line, box, tolerance=0):
    endpoints = line.get_start_and_end()
    x1, y1 = axes.p2c(endpoints[0])
    x2, y2 = axes.p2c(endpoints[1])    
    
    (xmin, ymin) = axes.p2c(box.get_corner(DL))
    (xmax, ymax) = axes.p2c(box.get_corner(UR))    
        
    x1 = round(x1, 3)
    x2 = round(x2, 3)
    y1 = round(y1, 3)
    y2 = round(y2, 3)
    xmin = round(xmin, 3)
    xmax = round(xmax, 3)
    ymin = round(ymin, 3)
    ymax = round(ymax, 3)

    intersections = []

    def add_intersection(t):
        x = round(x1 + t * (x2 - x1), 3)
        y = round(y1 + t * (y2 - y1), 3)
        if xmin - tolerance <= x <= xmax + tolerance and ymin - tolerance <= y <= ymax + tolerance:
            intersections.append((x,y))  # Round to avoid floating-point issues

    # Check intersection with left side (x = xmin)
    if x1 != x2:  
        t = (xmin - x1) / (x2 - x1)
        add_intersection(t)

    # Check intersection with bottom side (y = ymin)
    if y1 != y2:  
        t = (ymin - y1) / (y2 - y1)
        add_intersection(t)

    # Check intersection with right side (x = xmax)
    if x1 != x2:  
        t = (xmax - x1) / (x2 - x1)
        add_intersection(t)    

    # Check intersection with top side (y = ymax)
    if y1 != y2:  
        t = (ymax - y1) / (y2 - y1)
        add_intersection(t)
    
    intersections.sort()

    return intersections







# This scene is the intro without the fadeout,
# and doctored a bit for better screen appeal.
# Run this with -s in the manim CLI to get a png
# to use for the video thumbnail
class IntroImage(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            include_background_plane=False,
            include_foreground_plane=False
        )

    def construct(self):
        matrix = pematrix

        x = Vector(pex, color=TEAL)
        v = Vector(pev, color=PURPLE)
        circle = Circle(radius=1, color=BLUE)

        self.play(
            GrowArrow(x),
            GrowArrow(v),
            Create(circle)
        )
        for vec in [x, v]:
            self.add_vector(vec, animate=False)

        circles = SVD2Transpose(self, matrix, x, v, 1,
                                draw_scaling_vectors=False, circle1=circle)

        VGroup(circles, x, v).scale(1.35)

        headline = MathTex(r"(A^T ", r"\bar{v}", r")\cdot x = ",
                           r"\bar{v}", r"\cdot Ax", font_size=100).to_edge(DOWN)
        self.play(Write(headline))
        self.wait()

        title = Text("Understanding Matrix Transposes",
                     font_size=65).shift(UP*3)
        self.play(Write(title))

        self.wait(3)


##### Start Video Scenes #####

class Intro(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            include_background_plane=False,
            include_foreground_plane=False
        )

    def construct(self):
        matrix = pematrix

        x = Vector(pex, color=TEAL)
        v = Vector(pev, color=PURPLE)
        circle = Circle(radius=1, color=BLUE)

        self.play(
            GrowArrow(x),
            GrowArrow(v),
            Create(circle)
        )
        for vec in [x, v]:
            self.add_vector(vec, animate=False)

        circles = SVD2Transpose(self, matrix, x, v, 1,
                                draw_scaling_vectors=False, circle1=circle)

        headline = MathTex(r"(A^T ", r"\bar{v}", r")\cdot x = ", r"\bar{v}",
                           r"\cdot Ax", font_size=75).to_edge(DOWN).shift(UP*0.5)
        self.play(Write(headline))
        self.wait()

        title = Text("Understanding Matrix Transposes",
                     font_size=55).shift(UP*3)
        self.play(Write(title))
        self.wait(3)

        self.next_section()
        # Remove titles
        self.play(
            Unwrite(title),
            FadeOut(x),
            FadeOut(v),
            Uncreate(circles),
            Unwrite(headline)
        )


class IntroCard(Scene):
    def construct(self):
        text1 = Text("Transpose of a Matrix", font_size=60).shift(UP*0.5)
        text2 = MathTex("A^T", font_size=70).shift(DOWN*0.5)

        self.play(Write(text1))

        self.play(DrawBorderThenFill(text2))
        self.wait(30)

        self.play(Uncreate(VGroup(text1, text2)))
        self.wait()


class Prerequisites(Scene):
    def construct(self):

        # Text Prerequisites and underline
        prerequisites = Text("Prerequisites", font_size=36)
        prerequisites.set_width(1.5*prerequisites.get_width())
        prerequisites.move_to(UP*2)
        underline = Line(start=prerequisites.get_left()+DOWN*0.5+LEFT*0.25,
                         end=prerequisites.get_right()+DOWN*0.5+RIGHT*0.25, color=BLUE)

        # Show text
        self.play(Write(prerequisites), Create(underline))

        # List items
        items = [
            "Matrices as Linear Transformations",
            "Matrix Inverse and Identity Matrix",
            "Vector Dot Product",
            "Singular Value Decomposition (optional)"]
        item_list = BulletedList(*items)
        item_list.next_to(prerequisites, DOWN, buff=0.5)

        # Show list items one at a time
        for i in range(len(items)-1):
            self.play(Write(item_list[i]))
            self.wait(1.5)
        self.wait(3)
        self.play(Write(item_list[3]))

        self.wait(3)

        self.play(Uncreate(prerequisites), Uncreate(
            underline), Uncreate(item_list))


class Definition(Scene):
    def construct(self):
        # Draw and transpose 2x2 matrix, with flashing diagonal
        matrix = Matrix([[1, 2], [3, 4]]).scale(1.5)
        self.play(FadeIn(matrix), run_time=2.5)
        self.wait()
        transpose2(self, matrix, with_diagonal=True,
                   diagonal_flash=True, fade_when_done=True)
        self.wait()

        # Draw and transpose 3x3 matrix
        matrix = Matrix([['a_{1,1}', 'a_{1,2}', 'a_{1,3}'], [
                        'a_{2,1}', 'a_{2,2}', 'a_{2,3}'], ['a_{3,1}', 'a_{3,2}', 'a_{3,3}']]).scale(1.5)
        self.play(FadeIn(matrix), run_time=2.5)
        self.wait()
        transpose3(self, matrix, with_diagonal=True, fade_when_done=True)
        self.wait()

        # Transpose 3 by 2 matrix
        matrix = Matrix([['a_{1,1}', 'a_{1,2}'], ['a_{2,1}', 'a_{2,2}'], [
                        'a_{3,1}', 'a_{3,2}']]).scale(1.5)
        matrixt = Matrix([['a_{1,1}', 'a_{2,1}', 'a_{3,1}'], [
                         'a_{1,2}', 'a_{2,2}', 'a_{3,2}']]).scale(1.5)
        self.play(FadeIn(matrix), run_time=2.5)
        self.wait()
        transpose32(self, matrix, matrixt,
                    with_diagonal=True, fade_when_done=True)

        self.wait()


class Properties_Sum(Scene):
    def construct(self):
        # Formulas
        formula, underline = property_animation(self, "(A+B)^T", "=A^T+B^T")

        # Proof of sum of transpose
        # Matrices and symbols
        matrix1 = Matrix([["a", "b"], ["c", "d"]])
        matrix2 = Matrix([["e", "f"], ["g", "h"]])
        FixMatrixBaselines(matrix2, [0,1])
        matrix2.shift([0,GetBaseline(matrix1[0][0])-GetBaseline(matrix2[0][0]),0])
        plus_sign = MathTex("+")
        equals_sign = MathTex("=")
        matrix3 = Matrix([["a+e", "b+f"], ["c+g", "d+h"]], h_buff=1.7)
        FixMatrixBaselines(matrix3, [0,1])

        # Alignment, centered on matrix 2
        plus_sign.next_to(matrix2, LEFT)
        matrix1.next_to(plus_sign, LEFT)
        equals_sign.next_to(matrix2, RIGHT)
        matrix3.next_to(equals_sign, RIGHT)

        # Draw left side of equation
        self.play(Write(matrix1), Write(plus_sign),
                  Write(matrix2), Create(equals_sign))
        self.wait()

        # Add animation
        self.play(ReplacementTransform(matrix1.copy(), matrix3))
        self.wait()

        # Remove left side of equation
        self.play(
            FadeOut(matrix1), 
            FadeOut(plus_sign), 
            FadeOut(matrix2), 
            FadeOut(equals_sign), 
            matrix3.animate.move_to(matrix2))
        self.wait()

        # Transpose sum
        transpose2(self, matrix3)

        # Move to corner
        self.play(matrix3.animate.shift(UP*2.5+LEFT*5))

        # Other side of equation
        matrix1 = Matrix([["a", "b"], ["c", "d"]])
        matrix2 = Matrix([["e", "f"], ["g", "h"]])
        FixMatrixBaselines(matrix2, [0,1])
        matrix2.shift([0,GetBaseline(matrix1[0][0])-GetBaseline(matrix2[0][0]),0])
        plus_sign = MathTex("+")
        equals_sign = MathTex("=")
        matrix4 = Matrix([["a+e", "c+g"], ["b+f", "d+h"]], h_buff=1.7)
        FixMatrixBaselines(matrix4, [0,1])

        # Alignment, centered on matrix 2
        plus_sign.next_to(matrix2, LEFT)
        matrix1.next_to(plus_sign, LEFT)
        equals_sign.next_to(matrix2, RIGHT)
        matrix4.next_to(equals_sign, RIGHT)

        # Draw left side of equation
        self.play(Write(matrix1), Write(plus_sign),
                  Write(matrix2), Create(equals_sign))
        self.wait()

        # Transpose items
        transpose2(self, [matrix1, matrix2])

        # Add animation
        self.play(Write(equals_sign), ReplacementTransform(
            matrix1.copy(), matrix4))
        self.wait()

        # Remove left side of equation, move to corner
        self.play(
            FadeOut(matrix1), 
            FadeOut(plus_sign), 
            FadeOut(matrix2), 
            FadeOut(equals_sign), 
            matrix4.animate.shift(UP*2.5+RIGHT*1.25))
        self.wait()

        self.play(*[Unwrite(item)
                  for item in [matrix3, matrix4, formula, underline]])


class Properties_Product(Scene):
    def construct(self):
        # Formulas
        formula, underline = property_animation(self, "(AB)^T", "=B^T A^T")

        matrix1 = Matrix([["a", "b"], ["c", "d"]])
        matrix2 = Matrix([["e", "f"], ["g", "h"]])
        FixMatrixBaselines(matrix2, [0,1])
        matrix2.shift([0,GetBaseline(matrix1[0][0])-GetBaseline(matrix2[0][0]),0])
        matrix1.next_to(matrix2, LEFT)
        self.play(Write(matrix1), Write(matrix2))

        matrix3, equals_sign = multiply_matrix(self, matrix1, matrix2)

        # Remove left side of equation
        self.play(
            FadeOut(matrix1), 
            FadeOut(matrix2), 
            FadeOut(equals_sign), 
            matrix3.animate.move_to(matrix2))

        transpose2(self, matrix3)

        # Move to top left corner
        self.play(matrix3.animate.shift(UP*2.5+LEFT*4.7))

        # Rewrite matrices
        matrix1 = Matrix([["a", "b"], ["c", "d"]])
        matrix2 = Matrix([["e", "f"], ["g", "h"]])
        FixMatrixBaselines(matrix2, [0,1])
        matrix2.shift([0,GetBaseline(matrix1[0][0])-GetBaseline(matrix2[0][0]),0])
        matrix1.next_to(matrix2, LEFT)
        self.play(Write(matrix1), Write(matrix2))

        # Swap order
        self.play(matrix1.animate.move_to(matrix2),
                  matrix2.animate.move_to(matrix1))

        transpose2(self, [matrix2, matrix1])

        # Multiply. Note the arguments are supplied backwards to deal with the transpose-doesn't-re-assign-columns issue that I don't want to fix.
        matrix4 = Matrix([["ea+gb", "ec+gd"], ["fa+hb", "fc+hd"]], h_buff=2.15)
        matrix4, equals_sign = multiply_matrix(
            self, matrix1, matrix2, matrix4, location=matrix1)

        # Clear other side of equation
        self.play(
            FadeOut(matrix1), 
            FadeOut(matrix2), 
            FadeOut(equals_sign)
        )

        # Move to corner
        self.play(matrix4.animate.shift(UP*2.5+RIGHT*0.5))

        # Reorder entries to match other matrix
        entries = matrix4.get_entries()
        self.play(
            Transform(entries[0], MathTex("ae+bg").move_to(entries[0])),
            Transform(entries[1], MathTex("ce+dg").move_to(entries[1])),
            Transform(entries[2], MathTex("af+bh").move_to(entries[2])),
            Transform(entries[3], MathTex("cf+dh").move_to(entries[3]))
        )

        # Clear screen
        self.play(*[Unwrite(item)
                  for item in [matrix3, matrix4, formula, underline]])


class Properties_Repeat(Scene):
    def construct(self):
        # Formulas
        formula, underline = property_animation(self, "(A^T)^T", "=A^{T^T}=A")

        matrix1 = Matrix([["a", "b"], ["c", "d"]])
        self.play(Write(matrix1))

        transpose2(self, matrix1)
        transpose2(self, matrix1)

        # Tear down
        self.play(*[Unwrite(item) for item in [matrix1, formula, underline]])


class Properties_Inverse(Scene):
    def construct(self):
        formula, underline = property_animation(
            self, r"(A^{-1})^T", r" = (A^T)^{-1} ")
        run_time = 2

        formula1 = MathTex("(A^{-1})^T", font_size=75)
        self.play(Write(formula1), run_time=run_time)

        # Write equals and right side of equation
        formula2 = MathTex("(A^{-1})^T", "= ", "(A^{-1})^T", "I", font_size=75)
        self.play(
            ReplacementTransform(formula1[0], formula2[0]),
            Write(formula2[1]),
            TransformMatchingShapes(formula1[0].copy(), formula2[2]),
            Write(formula2[3]), run_time=run_time)

        # Expand I
        formula3 = MathTex("(A^{-1})^T", "= ", "(A^{-1})^T",
                           "A^T", "(A^T)^{-1}", font_size=75)
        self.play(
            ReplacementTransform(
                VGroup(formula2[0], formula2[1], formula2[2]),
                VGroup(formula3[0], formula3[1], formula3[2])),
            ReplacementTransform(formula2[3], VGroup(formula3[3], formula3[4])), run_time=run_time)

        # Collect into transpose
        formula4 = MathTex(
            "(A^{-1})^T", "= ", "(A", "A^{-1}", ")^T", "(A^T)^{-1}", font_size=75)
        self.play(
            ReplacementTransform(
                VGroup(formula3[0], formula3[1]),
                VGroup(formula4[0], formula4[1])),
            ReplacementTransform(formula3[2][0], formula4[2][0]),
            ReplacementTransform(formula3[3][0], formula4[2][1]),
            ReplacementTransform(formula3[3][1], formula4[4][1]),
            ReplacementTransform(
                VGroup(formula3[2][1], formula3[2][2], formula3[2][3]), formula4[3]),
            ReplacementTransform(
                VGroup(formula3[2][4], formula3[2][5]),
                formula4[4]),
            ReplacementTransform(formula3[4], formula4[5]), run_time=run_time)

        # Replace with Identity
        formula5 = MathTex("(A^{-1})^T", "= ", "(I)^T",
                           "(A^T)^{-1}", font_size=75)
        self.play(
            ReplacementTransform(
                VGroup(formula4[0], formula4[1], formula4[2][0]),
                VGroup(formula5[0], formula5[1], formula5[2][0])
            ),
            ReplacementTransform(
                VGroup(formula4[2][1], formula4[3]), formula5[2][1]
            ),
            TransformMatchingShapes(
                VGroup(formula4[4], formula4[5]),
                VGroup(formula5[2][2], formula5[2][3], formula5[3])
            ), run_time=run_time)

        # Get rid of I
        formula6 = MathTex("(A^{-1})^T", "= ", "", "(A^T)^{-1}", font_size=75)
        self.play(TransformMatchingTex(formula5, formula6), run_time=run_time)

        self.wait()

        # Tear down
        self.play(
            *[Unwrite(item) for item in
              [formula, underline, formula6]])
        self.wait()


class Properties_Dot(Scene):
    def construct(self):
        formula, underline = property_animation(self, "v \cdot x", "=v^T x")

        matrix1 = Matrix([["a"], ["b"]])
        matrix2 = Matrix([["c"], ["d"]])
        cdot = MathTex("\cdot")
        matrix1.next_to(cdot, LEFT)
        matrix2.next_to(cdot, RIGHT)

        # Write dot product
        self.play(Write(matrix1), Write(cdot), Write(matrix2))

        # Shift to the left and add equals
        self.play(VGroup(matrix1, matrix2, cdot).animate.shift(LEFT*2))

        # Do dot product
        result = MathTex("{{=}}{{ac}}", "{{+bd}}").next_to(matrix2)
        equals_sign = result.get_parts_by_tex("=")[0]
        ac = result.get_parts_by_tex("ac")[0]
        bd = result.get_parts_by_tex("+bd")[0]

        self.play(Write(equals_sign), run_time=0.3)

        source1 = VGroup(matrix1.get_entries()[0], matrix2.get_entries()[0])
        source2 = VGroup(matrix1.get_entries()[1], matrix2.get_entries()[1])
        self.play(Indicate(source1), TransformFromCopy(source1, ac))
        self.play(Indicate(source2), TransformFromCopy(source2, bd))

        # Switch sides of equation
        self.play(VGroup(matrix1, cdot, matrix2).animate.next_to(
            equals_sign, RIGHT), VGroup(ac, bd).animate.next_to(equals_sign, LEFT))

        matrix3 = Matrix([["a", "b"]]).next_to(equals_sign)
        self.play(matrix2.animate.next_to(matrix3),
                  ReplacementTransform(matrix1, matrix3), Uncreate(cdot))

        source1 = VGroup(matrix3.get_entries()[0], matrix2.get_entries()[0])
        source2 = VGroup(matrix3.get_entries()[1], matrix2.get_entries()[1])
        self.play(Indicate(source1), TransformFromCopy(source1, ac.copy()))
        self.play(Indicate(source2), TransformFromCopy(source2, bd.copy()))

        self.wait()

        # Tear down
        self.play(
            *[FadeOut(item) for item in self.mobjects])
        self.wait()


class IntroductoryTransformations(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = [[-1.4, 0.3], [2, 0.88]]
        x = Vector([1, 2], color=TEAL)

        # Draw axes
        self.remove(self.plane)
        self.remove(self.background_plane)
        self.play(Write(self.plane), Write(self.background_plane), run_time=2)
        self.wait()

        # add x vector
        self.add_vector(x)
        self.wait()

        # After many hours of hunting, I can only conclude that there
        # must be in a bug in Manim here. If you add the label after the vector,
        # then it will not stay fixed for the transformation, but if you add it before, it will.
        # Seems to be that it gets added to self.moving_mobjects for some reason.
        # The code below to remove it causes the scene to work as intended. Boo
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait(2)

        equations = MathTex(r"{{\bar{x}=Ax}}",
                            r"{{\\ \bar{v}=Av}}", font_size=70)
        equations[0][1].set_color(TEAL)
        equations[0][4].set_color(TEAL)
        equations[1][1].set_color(PURPLE)
        equations[1][4].set_color(PURPLE)
        equations.to_edge(UR)
        equations[0].add_background_rectangle(opacity=0)
        equations[1].add_background_rectangle(opacity=0)
        self.play(
            Write(equations[0]), equations[0].background_rectangle.animate.set_opacity(0.75))
        self.wait()

        # Un-transform
        self.moving_mobjects = []
        self.apply_inverse(matrix, run_time=2)
        self.wait()

        # Add second vector
        v = Vector([-1, 2], color=PURPLE)
        self.add_vector(v)
        self.wait()

        # Transform again
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()

        # Add second equation
        self.play(
            Write(equations[1]), equations[1].background_rectangle.animate.set_opacity(0.75))
        self.wait()

        self.moving_mobjects = []
        self.apply_inverse(matrix, run_time=2)
        self.wait()

        dot_product = MathTex(r"{{v\cdot x}}", r"\stackrel{?}{=}", r"\bar{v}\cdot \bar{x}",
                              font_size=100, stroke_color='#000000', stroke_width=0.5).shift(DOWN*1.7+LEFT*3.5)
        dot_product[0][0].set_color(PURPLE)
        dot_product[2][1].set_color(PURPLE)
        dot_product[0][2].set_color(TEAL)
        dot_product[2][4].set_color(TEAL)
        orig_product = dot_product[0].add_background_rectangle()
        equals_sign = dot_product[1].add_background_rectangle(opacity=0)
        trans_product = dot_product[2].add_background_rectangle(opacity=0)

        # Write first bit of formula
        self.play(FadeIn(orig_product))
        self.wait()
        # Transform again
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()
        # Write rest of formula
        self.play(FadeIn(equals_sign), FadeIn(trans_product), equals_sign.background_rectangle.animate.set_opacity(
            0.75), trans_product.background_rectangle.animate.set_opacity(0.75))
        self.wait(3)

        self.play(*[FadeOut(item) for item in [self.plane,
                  self.background_plane, x, v, equations]])
        self.play(dot_product.animate.center(), run_time=1.5)
        self.play(*[
            dot_product[0][1].animate.set_color(WHITE),
            dot_product[2][2].animate.set_color(WHITE),
            dot_product[0][3].animate.set_color(WHITE),
            dot_product[2][5].animate.set_color(WHITE)
        ], run_time=2)

        self.wait()


class NotEqualProof(Scene):
    def construct(self):
        formula1 = MathTex(
            r"v\cdot x", r"\stackrel{?}{=}", r"\bar{v}\cdot \bar{x}", font_size=100)

        # Display whole equation, which should match end of previous scene.
        self.add(formula1)
        self.wait()

        self.play(formula1[0].animate.shift(
            LEFT), formula1[2].animate.shift(RIGHT), Unwrite(formula1[1]))
        self.wait()

        fl = MathTex(r"v\cdot x", font_size=100).move_to(formula1[0])
        fr = MathTex(r"\bar{v}\cdot", r"\bar{x}",
                     font_size=100).move_to(formula1[2])
        self.add(fl, fr)
        self.remove(formula1[0], formula1[1], formula1[2], formula1)

        # Convert dot product to transposes
        fl2 = MathTex(r"v^T x", font_size=100).move_to(fl).align_to(fl, DOWN)
        self.play(Transform(fl, fl2))
        self.wait()

        fr2 = MathTex(r"\bar{v}^T", r"\bar{x}", font_size=100).move_to(
            fr).align_to(fr, DOWN)
        self.play(Transform(fr, fr2))
        self.wait()

        # Substitute
        fr3 = MathTex(r"(Av)^T", "(Ax)", font_size=100).move_to(
            fr).align_to(fl, UP)
        self.play(TransformMatchingShapes(fr, fr3), run_time=2)
        self.wait(2)

        # Distribute transpose into product, remove parenthesis
        fr4 = MathTex("v^T A^T", "(Ax)", font_size=100).move_to(
            fr3).align_to(fr3, UP)
        self.play(TransformMatchingShapes(
            fr3[0], fr4[0]), ReplacementTransform(fr3[1], fr4[1]), run_time=2)
        self.wait()

        # Rearrange parenthesis
        fr5 = MathTex("v^T", "(A^T A)", "x", font_size=100).move_to(
            fr4).align_to(fr4, UP)
        self.play(TransformMatchingShapes(fr4, fr5), run_time=2)
        self.wait(2)

        # Flash ATA
        self.play(
            Indicate(VGroup(fr5[1][1], fr5[1][2], fr5[1][3])), run_time=2)
        self.wait()

        # Write conclusion
        text = Tex(r"$v\cdot x$ ", r" $=$ ", r" $\bar{v}\ \cdot $ ",
                   r" $\bar{x}$ ", r"\\ if ", r" $A^T A = I$", font_size=85).shift(UP*2.5)
        anims = [
            TransformMatchingShapes(fl.copy(), text[0]),
            Write(text[1]),
            TransformMatchingShapes(
                VGroup(fr5[0].copy(), fr5[2].copy()), VGroup(text[2], text[3])),
            Write(text[4]),
            TransformMatchingShapes(fr5[1].copy(), text[5])
        ]
        self.play(*anims, run_time=2)
        self.wait(3)

        # Quick demonstration
        fr6 = MathTex(r"v^T", "A^T A", "x", font_size=100).move_to(
            fr5).align_to(fr5, UP)
        fr7 = MathTex(r"v^T", "I", "x", font_size=100).move_to(
            fr6).align_to(fr6, UP)
        fr8 = MathTex(r"v^T", "", "x", font_size=100).move_to(
            fr7).align_to(fr7, UP)
        self.play(TransformMatchingShapes(fr5, fr6))
        self.play(TransformMatchingShapes(fr6, fr7),
                  TransformMatchingShapes(text[5][4].copy(), fr7[1]))
        self.play(TransformMatchingShapes(fr7, fr8))
        self.wait(4)

        # Remove side expressions, center conclusions
        textc = text.copy().center().set(font_size=100)
        self.play(FadeOut(fl), FadeOut(fr8),
                  Transform(text, textc), run_time=1.5)
        self.wait()

        # Inverse = transpose
        text5 = MathTex(
            "A^T = A^{-1}", font_size=85).move_to(text[5]).align_to(text[5], DOWN)
        self.play(TransformMatchingShapes(
            text[5], text5), run_time=2, path_arc=-PI/2)
        self.wait(8)

        # Conclusion
        text6 = Tex(
            r"Linear transformations do not \\ necessarily preserve the dot product", font_size=60)
        text6.next_to(text5, DOWN)
        text6.move_to(np.multiply(text6.get_center(), [0, 1, 1]))
        self.play(Write(text6))
        self.wait()

        self.play(*[Unwrite(text[i])
                  for i in range(5)], Unwrite(text5), Unwrite(text6))
        self.wait()


class DotProductNotInvariantDemo(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        # Draw axes
        self.remove(self.plane)
        self.remove(self.background_plane)
        self.play(FadeIn(self.plane), FadeIn(
            self.background_plane), run_time=2)
        self.wait()

        equation = MathTex(r"v \cdot x = \|v\|\|x\|cos\theta", font_size=70)
        equation.to_edge(UR).add_background_rectangle()
        self.play(Write(equation))
        self.wait(5)

        matrix = [[1, -2], [0, 1]]
        x = Vector([2, 0], color=MAROON)
        v = Vector([2, 1], color=GOLD)

        self.add_vector(x)
        self.add_vector(v)

        self.wait(4)

        self.moving_mobjects = []
        self.apply_matrix(matrix)

        self.wait(4)

        self.play(*[ShrinkToCenter(item) for item in [self.plane,
                  self.background_plane, equation]], Uncreate(x), Uncreate(v))
        self.remove(*[ShrinkToCenter(item)
                    for item in [self.plane, self.background_plane, equation, x, v]])

        self.wait()


class DeriveM(Scene):
    def construct(self):
        self.wait()
        xt = MathTex(r"\bar{x}", "=", "Ax", font_size=100)
        xt1 = xt.copy().shift(LEFT*2.5)
        vt = MathTex(r"\bar{v}", "=", "A", "v", font_size=100).shift(RIGHT*2.5)

        # Display x->Ax
        self.play(Write(xt))
        self.wait()

        # Split to two formulas
        self.play(Transform(xt, xt1), TransformMatchingShapes(
            xt.copy(), vt), run_time=1.5)
        self.wait(2)

        # Change A to M
        vm = MathTex(r"\bar{v}", "=", "M", "v", font_size=100).shift(RIGHT*2.5)
        self.play(Transform(vt, vm))
        self.play(Circumscribe(vt[2][0]), run_time=1.5)
        self.wait(2)

        # shift transforms up and add dot product equation
        dp = MathTex("v^T x", "=", r"\bar{v}^T \bar{x}", font_size=100)
        self.play(xt.animate.shift(UP*2), vt.animate.shift(UP*2))
        self.play(Write(dp))
        self.wait(2)

        # Substitute
        dpr = MathTex("(Mv)^T", "(Ax)", font_size=100).next_to(
            dp[1]).align_to(dp[0], UP)
        self.play(
            TransformMatchingShapes(dp[2], dpr),
            ReplacementTransform(xt[2].copy(), VGroup(dpr[1][1], dpr[1][2])),
            ReplacementTransform(vt[2].copy(), VGroup(dpr[0][1], dpr[0][2])), run_time=2)
        # self.wait()

        # Distribute transpose
        dprt = MathTex("v^T M^T", font_size=100).next_to(
            dpr[1], LEFT).align_to(dp[0], DOWN)
        self.play(TransformMatchingShapes(dpr[0], dprt), run_time=2)
        # self.wait()

        # Move parenthesis
        dprp = MathTex("v^T", "(M^T A)", "x", font_size=100).move_to(
            dpr).align_to(dpr, UP)
        self.play(TransformMatchingShapes(
            VGroup(dprt, dpr[1]), dprp), run_time=2)
        # self.wait()

        # MTA = I
        mta = MathTex("M^T A", "=", "I", font_size=100)
        self.play(
            TransformMatchingShapes(dprp[1], mta[0]),
            ReplacementTransform(dp[1], mta[1]),
            Write(mta[2]),
            FadeOut(dp[0]),
            FadeOut(dprp[0]),
            FadeOut(dprp[2]), run_time=2)
        self.wait(1)

        # MT = A-1
        mit = MathTex("M^T = A^{-1}", font_size=100)
        self.play(TransformMatchingShapes(
            mta, mit, path_arc=PI/2), run_time=1.5)
        self.wait(2)

        # M = A-tT
        mdef = MathTex("M = ", "(A^{-1})^T", font_size=100)
        self.play(TransformMatchingShapes(
            mit, mdef, path_arc=-PI/2), run_time=1.5)
        self.play(Circumscribe(mdef[1]), run_time=1.5)
        self.wait(4)

        # Substitute result back into formula, and center things
        vt2 = MathTex(r"\bar{v}", "=", "(A^{-1})^T", "v",
                      font_size=100).align_to(vt, DL)
        vt2.shift(vt[0].get_center()-vt2[0].get_center())
        sl = VGroup(xt, vt2).get_center()[0]
        vt2.shift(LEFT*sl)
        self.play(
            xt.animate.shift(LEFT*sl),
            ReplacementTransform(
                VGroup(vt[0], vt[1], vt[3]),
                VGroup(vt2[0], vt2[1], vt2[3])),
            FadeOut(vt[2]),
            TransformFromCopy(mdef[1], vt2[2]), run_time=2)
        self.wait(5)

        # Add qualifier
        paren = Tex(r"($=A$ if $A^{-1} = A^T)$").shift(DOWN*1.5)
        self.play(FadeIn(paren))
        self.wait(5)

        # Remove everything
        self.play(*[Uncreate(item, run_time=5) for item in [xt, vt2, mdef, paren]])
        self.wait()


class TransformWithM1(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = pematrix
        x = Vector(pex, color=TEAL)
        v = Vector(pev, color=PURPLE)

        # draw axes
        self.remove(self.background_plane)
        self.remove(self.plane)
        self.play(GrowFromCenter(self.background_plane),
                  GrowFromCenter(self.plane), run_time=3)

        self.add_vector(x)
        self.add_vector(v)

        # Add equations
        xeqn = MathTex(r"\bar{x}", "= A", "x", font_size=65).to_edge(
            UR).add_background_rectangle()
        veqn = MathTex(r"\bar{v}", "=", "A", "v", font_size=65).to_edge(
            UR).shift(DOWN).add_background_rectangle()
        xeqn[1][1].set_color(TEAL)
        xeqn[3].set_color(TEAL)
        veqn[1][1].set_color(PURPLE)
        veqn[4].set_color(PURPLE)
        self.play(Write(xeqn), Write(veqn))

        # First transform
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()

        # Reverse it
        self.moving_mobjects = []
        self.apply_inverse(matrix)
        self.wait()

        # Transform again
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait(2)

        # Dash x
        xghost = DashedVMobject(x, num_dashes=10)
        self.play(FadeTransform(
            x, xghost, replace_mobject_with_target_in_scene=True))

        # Reverse transform, but leave x behind
        self.moving_mobjects = []
        self.moving_vectors.remove(x)
        self.apply_inverse(matrix)
        self.wait(4)

        # Transform equation
        veqn2 = MathTex(r"\bar{v}", "=", "A^{-1^T}", "v",
                        font_size=65).align_to(veqn, DR).add_background_rectangle()
        veqn2[1][1].set_color(PURPLE)
        veqn2[4].set_color(PURPLE)
        self.play(ReplacementTransform(veqn, veqn2))
        self.wait(2)

        # Transform by M
        self.moving_mobjects = []
        self.apply_inverse_transpose(matrix)
        self.wait(3)

        # Dash v
        vghost = DashedVMobject(v, num_dashes=5)
        self.add(vghost)

        # Return original vectors
        self.moving_mobjects = []
        self.apply_transposed_matrix(matrix)
        xcopy = Vector(pex, color=TEAL)
        self.add_vector(xcopy)
        self.wait(2)

        # Indicate solids then dashes
        self.play(
            ApplyWave(xcopy),
            ApplyWave(v),
            ApplyWave(xeqn[3]),
            ApplyWave(veqn2[4])
        )
        self.wait(0.5)
        self.play(
            ApplyWave(xghost),
            ApplyWave(vghost),
            ApplyWave(xeqn[1]),
            ApplyWave(veqn2[1])
        )
        self.wait(3)

        # Remove vectors
        self.play(*[Uncreate(item) for item in [
            xcopy, xghost, v, vghost, xeqn, veqn2]])

        self.wait()


class TransformWithM2(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = [[0, 1], [-1, 0]]
        x = Vector([-1, 1], color=YELLOW)
        v = Vector([0.5, 2], color=ORANGE)

        self.play(GrowArrow(x))
        self.play(GrowArrow(v))
        self.wait(0.5)

        xghost = (DashedVMobject(x, num_dashes=8))
        self.add(xghost)
        self.add_transformable_mobject(xghost)
        vghost = (DashedVMobject(v, num_dashes=10))
        self.add(vghost)
        self.add_transformable_mobject(vghost)

        # transform
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait(4)

        # Indicate solids then dashes
        self.play(ApplyWave(x), ApplyWave(v))
        self.wait(0.5)
        self.play(ApplyWave(xghost), ApplyWave(vghost))
        self.wait(2)

        # fade out
        self.play(FadeOut(*self.mobjects), run_time=2)
        self.wait()


class Orthogonal(Scene):
    def construct(self):
        self.wait(2)

        # t = -1
        formula = MathTex(r"A^T=A^{-1}", font_size=75)
        self.play(Write(formula), run_time=2)
        self.wait(2)

        label = Text("Orthogonal Matrix").shift(UP)
        underline = Line(label.get_left()+DOWN*0.5+LEFT*0.25,
                         label.get_right()+DOWN*0.5+RIGHT*0.25, color=BLUE)
        self.play(Write(label), formula.animate.shift(DOWN*0.5))
        self.play(Create(underline))
        self.wait(6)

        bullet = Text("Preserves Dot Products").next_to(
            formula, DOWN).shift(DOWN*0.5).scale(0.87)
        self.play(Write(bullet))
        self.wait(6)

        self.play(Uncreate(VGroup(formula, label, underline, bullet)))
        self.wait()


class OrthogonalDemo(LinearTransformationScene):
    def construct(self):
        # draw axes and basis vectors
        self.remove(self.background_plane)
        self.remove(self.plane)
        self.remove(self.basis_vectors)
        self.play(FadeIn(self.background_plane),
                  FadeIn(self.plane), run_time=2)
        self.play(GrowArrow(self.basis_vectors[0]))
        self.play(GrowArrow(self.basis_vectors[1]))

        text = Tex("Orthogonal Transformation: ", "Reflection",
                   font_size=60).to_corner(UR).add_background_rectangle()
        self.play(Write(text))

        # reflection
        matrix1 = [[1, 0], [0, -1]]
        self.moving_mobjects = []
        self.apply_matrix(matrix1)
        self.wait()

        # rotation
        text2 = Tex("Orthogonal Transformation: ", "Rotation",
                    font_size=60).to_corner(UR).add_background_rectangle()
        self.play(Transform(text, text2))
        matrix2 = [[1/sqrt(2), -1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]]
        self.moving_mobjects = []
        self.apply_matrix(matrix2)
        self.wait(7)

        # undo transformations
        text3 = Tex("Orthogonal Transformation: ", "Transpose = Inverse",
                    font_size=60).to_corner(UR).add_background_rectangle()
        self.play(Transform(text, text3))
        self.wait()
        self.moving_mobjects = []
        self.apply_transposed_matrix(matrix2)
        self.moving_mobjects = []
        self.apply_transposed_matrix(matrix1)
        self.wait(2)

        self.play(*[FadeOut(item) for item in self.mobjects], run_time=3)
        self.wait()


class SVDIntro(Scene):
    def construct(self):
        self.wait()

        text = Text("Singular Value Decomposition (SVD)").scale(0.85).shift(UP)
        self.play(Write(text))
        underline = Line(text.get_left() + DOWN*0.5+LEFT*0.25,
                         text.get_right()+DOWN*0.5+RIGHT*0.25).set_color(BLUE)
        self.play(Create(underline))
        self.wait(5)

        list_items = ["Rotate/Reflect", "Scale at the Axes", "Rotate/Reflect"]
        list_items_mob = VGroup(*[Tex(f"{i}. {item}") for i, item in enumerate(
            list_items, start=1)]).arrange(DOWN, aligned_edge=LEFT).next_to(underline, DOWN)
        # crazy code to fix the weird vertical spacing problem
        list_items_mob[2].shift([0, ((list_items_mob[1][0][1].get_center()[1]-list_items_mob[0][0][1].get_center()[
                                1])-(list_items_mob[2][0][1].get_center()[1]-list_items_mob[1][0][1].get_center()[1])), 0])
        for item in list_items_mob:
            self.play(Create(item), run_time=1.5)
            self.wait()
        self.wait(2)

        screen = Rectangle(BLUE_E, 15, 15).set_fill(BLUE_E, 1)
        self.play(
            FadeIn(screen),
            *[item.animate.set_color(BLUE_E) for item in self.mobjects], run_time=2.5)
        self.wait()


class SVDDemo(LinearTransformationScene):
    def construct(self):
        screen = Rectangle(BLUE_E, 15, 15).set_fill(BLUE_E, 1)
        self.add(screen)
        self.play(FadeOut(screen), run_time=2.5)

        matrix = np.array([
            [2, 2],
            [0.3, 1.7]
        ])

        # Original transformation
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait(2)

        # Leave ghost vectors (loops are workaround for arrow bug: https://github.com/ManimCommunity/manim/issues/3220)
        xghost = DashedVMobject(
            Vector(np.dot(matrix, [1, 0]), color=GREEN), num_dashes=8).set_opacity(0.35)
        for i in range(len(xghost)-1):
            xghost[i].remove(xghost[i][-1])
        vghost = DashedVMobject(
            Vector(np.dot(matrix, [0, 1]), color=RED), num_dashes=10).set_opacity(0.35)
        for i in range(len(vghost)-1):
            vghost[i].remove(vghost[i][-1])
        self.add(xghost), self.add(vghost)

        # Restore to original plane
        self.moving_mobjects = []
        self.apply_matrix(np.linalg.inv(matrix), run_time=2)
        self.wait()

        # SVD
        circle = SVDAnim(self, matrix)
        self.wait()

        # Remove stuff
        self.play(
            *[FadeOut(item) for item in self.mobjects], run_time=3)
        self.wait()


class SVDbreakdown(Scene):
    def construct(self):
        self.wait()

        # Draw SVD
        svd = MathTex("A", "=", "R_2", "\Sigma", "R_1", font_size=80)
        a = svd[0].copy().center()
        self.play(Write(a))
        self.play(ReplacementTransform(a, svd[0]))
        self.play(Write(svd[1]))
        self.play(TransformFromCopy(svd[0], svd[4]), run_time=1.5)
        self.play(TransformFromCopy(svd[0], svd[3]), run_time=1.5)
        self.play(TransformFromCopy(svd[0], svd[2]), run_time=1.5)
        self.wait(3)

        # Show standard notation, then switch back
        svdn = MathTex("A", "=", "U", "\Sigma", "V^T",
                       font_size=80).align_to(svd, DOWN)
        self.play(TransformMatchingTex(svd, svdn))
        self.wait()
        self.play(Indicate(svdn[2]), Indicate(svdn[4]))
        self.wait(2)
        self.play(TransformMatchingTex(svdn, svd))
        self.wait(8)

        # Shift up SVD, add formulas
        self.play(Transform(svd, svd.copy().shift(UP*3).set_font_size(70)))
        xt = MathTex(r"\bar{x} =", "A", "x", font_size=70).shift(LEFT*3.5+DOWN)
        vt = MathTex(r"\bar{v} =", r"A^{-1^T}", "v",
                     font_size=70).shift(RIGHT*3.5+DOWN).align_to(xt, DOWN)
        self.play(DrawBorderThenFill(xt), DrawBorderThenFill(vt))
        self.wait()

        # Substitute SVD in x
        xtsvd = MathTex(r"\bar{x}=", "(", r"R_2", r"\Sigma",
                        r"R_1", ")", "x", font_size=70).shift(LEFT*3.5+DOWN)
        self.play(
            TransformMatchingShapes(xt[0], xtsvd[0]),
            ReplacementTransform(xt[1], VGroup(xtsvd[1], xtsvd[5])),
            TransformFromCopy(VGroup(svd[2], svd[3], svd[4]), VGroup(
                xtsvd[2], xtsvd[3], xtsvd[4])),
            TransformMatchingShapes(xt[2], xtsvd[6]), run_time=1.5)
        self.wait()

        # Substitute svd into v
        vtsvd = MathTex(r"\bar{v}=", "(", r"R_2", r"\Sigma", r"R_1", r")^{-1^T}",
                        "v", font_size=70).shift(RIGHT*3.5+DOWN).align_to(xtsvd, DOWN)
        self.play(
            TransformMatchingShapes(vt[0], vtsvd[0]),
            TransformMatchingShapes(vt[1], VGroup(vtsvd[1], vtsvd[5])),
            TransformFromCopy(VGroup(svd[2], svd[3], svd[4]), VGroup(
                vtsvd[2], vtsvd[3], vtsvd[4])),
            TransformMatchingShapes(vt[2], vtsvd[6]), run_time=1.5)
        self.wait()

        # Distribute inverse
        # First swap order then distribute
        vtinv0 = MathTex(r"\bar{v}=", "(", r"R_1", r"\Sigma", r"R_2", r")^{-1^T}",
                         "v", font_size=70).shift(RIGHT*3.5+DOWN).align_to(xtsvd, DOWN)
        self.play(
            TransformMatchingTex(vtsvd, vtinv0), run_time=1.5)

        vtinv = MathTex(r"\bar{v}=", "(", r"R_1^{-1}", r"\Sigma^{-1}", r"R_2^{-1}",
                        r")^{T}", "v", font_size=70).shift(RIGHT*3.5+DOWN).align_to(xtsvd, DOWN)
        self.play(
            ReplacementTransform(vtinv0[0], vtinv[0]),
            ReplacementTransform(vtinv0[1], vtinv[1]),
            TransformMatchingShapes(vtinv0[2][0], vtinv[2][0]),
            TransformMatchingShapes(vtinv0[2][1], vtinv[2][3]),
            TransformMatchingShapes(vtinv0[3][0], vtinv[3][0]),
            TransformMatchingShapes(vtinv0[4][0], vtinv[4][0]),
            TransformMatchingShapes(vtinv0[4][1], vtinv[4][3]),
            TransformMatchingShapes(vtinv0[5][1].copy(), vtinv[2][1]),
            TransformMatchingShapes(vtinv0[5][2].copy(), vtinv[2][2]),
            TransformMatchingShapes(vtinv0[5][1].copy(), vtinv[3][1]),
            TransformMatchingShapes(vtinv0[5][2].copy(), vtinv[3][2]),
            TransformMatchingShapes(vtinv0[5][1], vtinv[4][1]),
            TransformMatchingShapes(vtinv0[5][2], vtinv[4][2]),
            ReplacementTransform(vtinv0[5][0], vtinv[5][0]),
            ReplacementTransform(vtinv0[5][3], vtinv[5][1]),
            TransformMatchingShapes(vtinv0[6], vtinv[6]), run_time=1.5)
        self.wait(5)

        # Properites of SVD, sigma
        sigmat0 = MathTex(r"\Sigma^T", font_size=70).shift(UP*1+LEFT*4)
        sigmat1 = MathTex(r"\Sigma^T", "=", r"\Sigma",
                          font_size=70).align_to(sigmat0, LEFT+DOWN)
        self.play(Write(sigmat0))
        self.wait(1.5)
        self.play(
            ReplacementTransform(sigmat0[0], sigmat1[0]),
            Write(sigmat1[1]))
        self.play(
            TransformFromCopy(sigmat0[0][0], sigmat1[2]), run_time=1.5)
        self.wait(2)
        sigmat2 = MathTex(r"\Sigma^{-1^T}", "=", r"\Sigma^{-1}",
                          font_size=70).move_to(sigmat1).align_to(sigmat1, DOWN)
        self.play(
            ReplacementTransform(sigmat1[0][0], sigmat2[0][0]),
            ReplacementTransform(sigmat1[1], sigmat2[1]),
            ReplacementTransform(sigmat1[2], sigmat2[2][0]),
            ReplacementTransform(sigmat1[0][1], sigmat2[0][3]),
            Write(sigmat2[0][1]), Write(sigmat2[0][2]),
            Write(sigmat2[2][1]), Write(sigmat2[2][2]), run_time=1.5)
        self.wait(3)

        # Properites of SVD, R
        rt0 = MathTex(r"R^T", font_size=70).align_to(
            sigmat2, DOWN).shift(2.5*RIGHT)
        rt1 = MathTex(r"R^T", "=", "R^{-1}", font_size=70).align_to(rt0, DL)
        self.play(Write(rt0))
        self.wait()
        self.play(
            ReplacementTransform(rt0[0], rt1[0]),
            Write(rt1[1]))
        self.play(
            TransformFromCopy(rt0[0][0], rt1[2]), run_time=1.5)
        self.wait(2)
        rt2 = MathTex(r"R^{-1^T}", "=", "R", font_size=70).align_to(rt1, DL)
        self.play(
            ReplacementTransform(rt1[0][0], rt2[0][0]),
            ReplacementTransform(rt1[0][1], rt2[0][3]),
            ReplacementTransform(rt1[1], rt2[1]),
            ReplacementTransform(rt1[2][0], rt2[2]),
            ReplacementTransform(rt1[2][1], rt2[0][1], path_arc=PI/3),
            ReplacementTransform(rt1[2][2], rt2[0][2], path_arc=PI/3), run_time=1.5)
        self.wait(3)

        # Distribute transpose
        vttr0 = MathTex(r"\bar{v}=", "(", r"R_2^{-1}", r"\Sigma^{-1}", r"R_1^{-1}",
                        r")^{T}", "v", font_size=70).shift(RIGHT*3.5).align_to(xtsvd, DOWN)
        self.play(TransformMatchingTex(vtinv, vttr0), run_time=2)

        vttr = MathTex(r"\bar{v}=", "(", r"R_2", r"\Sigma^{-1}", r"R_1", r")",
                       "v", font_size=70).shift(RIGHT*3).align_to(xtsvd, DOWN)

        self.play(
            ReplacementTransform(vttr0[0], vttr[0]),
            ReplacementTransform(vttr0[1], vttr[1]),
            ReplacementTransform(vttr0[5], vttr[5]),
            ReplacementTransform(vttr0[6], vttr[6]),
            ReplacementTransform(
                VGroup(vttr0[2][0], vttr0[2][3]), VGroup(vttr[2][0], vttr[2][1])),
            ReplacementTransform(
                VGroup(vttr0[4][0], vttr0[4][3]), VGroup(vttr[4][0], vttr[4][1])),
            FadeOut(VGroup(vttr0[2][1], vttr0[2]
                    [2], vttr0[4][1], vttr0[4][2])),
            ReplacementTransform(vttr0[3], vttr[3]),
            ReplacementTransform(rt2[2].copy(), vttr[2][0]),
            ReplacementTransform(rt2[2].copy(), vttr[4][0]),
            ReplacementTransform(sigmat2[2].copy(), vttr[3]), run_time=2)
        self.wait(7)

        # Flash sections of svds
        self.play(
            Indicate(xtsvd[4]),
            Indicate(vttr[4]), run_time=2.5)
        self.play(
            Indicate(xtsvd[3]),
            Indicate(vttr[3], color=BLUE), run_time=2.5)
        self.play(
            Indicate(xtsvd[2]),
            Indicate(vttr[2]), run_time=2.5)
        self.wait(2)

        self.play(
            *[FadeOut(item) for item in self.mobjects], run_time=3)

        self.wait()


class SVDdot1(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        # draw things
        self.remove(self.plane)
        self.remove(self.background_plane)
        self.play(FadeIn(VGroup(self.plane, self.background_plane)), run_time=3)

        matrix = np.array(pematrix)
        u, s, vt = ComputeSVD(matrix)

        nx = np.array(pex)
        nv = np.array(pev)
        x = Vector(nx, color=TEAL)
        v = Vector(nv, color=PURPLE)
        vghost = Vector(nv, color=PURPLE)

        self.add_vector(x)

        # Original transformation
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()

        # Restore to original plane
        self.moving_mobjects = []
        self.apply_matrix(np.linalg.inv(matrix))
        self.wait()

        # Create circle
        circle1 = Circle(radius=1).set_color(BLUE)
        self.play(Create(circle1))
        self.add_transformable_mobject(circle1)
        self.wait()

        # SVD with x
        SVDAnim(self, matrix, circle=circle1)
        self.wait()

        # Restore but leave x
        xghost = DashedVMobject(x, num_dashes=10)
        self.moving_mobjects = []
        self.apply_inverse(matrix, added_anims=[FadeTransform(x, xghost)])
        self.moving_vectors.remove(x)
        self.wait()

        # Add v vector, and equation
        self.add_vector(v)
        self.wait()

        # Apply SVD transformation 1
        SVDAnim(self, matrix, circle1, invert_s=True)

        # Switch to vghost
        vghost = DashedVMobject(v, num_dashes=5)
        self.play(FadeTransform(v, vghost))
        self.wait()

        # Restore, remove basis and vectors, and plane, and equations
        self.moving_vectors.remove(v)
        self.transformable_mobjects.remove(self.plane)
        self.moving_mobjects = []
        self.apply_transposed_matrix(matrix, added_anims=[
            FadeOut(self.plane),
            Unwrite(xghost), Unwrite(vghost)
        ])

        # Re-add original vectors
        x = self.add_vector(nx, color=TEAL)
        v = self.add_vector(nv, color=PURPLE)

        circles = SVD2Transpose(self, matrix, x, v, circle1=circle1)
        self.wait()

        # Tear down
        self.play(
            *[Uncreate(item) for item in [
                circles,
                x, v
            ]], run_time=2)

        self.wait()


class SVDdot2(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        # starting place
        self.remove(self.plane)
        self.transformable_mobjects.remove(self.plane)

        # constants n' such
        matrix = MatrixFromSVD(PI/3, [0.5, 1.3], -PI/9)
        u, s, vt = ComputeSVD(matrix)

        nx = np.array([1, 1.1])
        nv = np.array([1.3, -0.2])
        x = Vector(nx, color=PINK)
        v = Vector(nv, color=GREEN)

        # x vector text
        atex0 = MathTex(r"\bar{x}=", "A", "x", font_size=65).to_edge(
            UR).add_background_rectangle()
        atex = MathTex(r"\bar{x}=", "R_2", r"\Sigma", "R_1", "x", font_size=65).to_edge(
            UR).add_background_rectangle()
        atex0[1][1].set_color(PINK)
        atex0[3].set_color(PINK)
        atex[1][1].set_color(PINK)
        atex[5].set_color(PINK)

        # eqn animation
        self.add_vector(x)
        self.play(Write(atex0))
        self.play(*[
            ReplacementTransform(atex0[0], atex[0]),
            TransformMatchingShapes(atex0[1], atex[1]),
            TransformMatchingShapes(atex0[3], atex[5]),
            ReplacementTransform(
                atex0[2],
                VGroup(atex[2], atex[3], atex[4])
            )
        ], run_time=2)

        # v equation
        vtex0 = MathTex(r"\bar{v}=", "A^{-1^T}", "v",
                        font_size=65).to_edge(UR).shift(DOWN).add_background_rectangle()
        vtex = MathTex(r"\bar{v}=", "R_2", r"\Sigma^{-1}", "R_1", "v",
                       font_size=65).to_edge(UR).shift(DOWN).add_background_rectangle()
        vtex0[1][1].set_color(GREEN)
        vtex0[3].set_color(GREEN)
        vtex[1][1].set_color(GREEN)
        vtex[5].set_color(GREEN)

        # Add v vector, and equation
        self.add_vector(v)
        self.play(Write(vtex0))
        self.play(*[
            ReplacementTransform(vtex0[0], vtex[0]),
            TransformMatchingShapes(vtex0[1], vtex[1]),
            TransformMatchingShapes(vtex0[3], vtex[5]),
            ReplacementTransform(
                vtex0[2],
                VGroup(vtex[2], vtex[3], vtex[4])
            )
        ], run_time=2)
        self.wait()

        # Create circle
        circle1 = Circle(radius=1).set_color(BLUE)

        #  do eqn and draw circle
        self.play(Create(circle1))
        self.add_transformable_mobject(circle1)
        self.wait()

        circles = SVD2Transpose(self, matrix, x, v, circle1=circle1, added_anims=[
            [Indicate(atex[4]), Indicate(vtex[4])],
            [Indicate(atex[3]), Indicate(vtex[3], color=BLUE)],
            [Indicate(atex[2]), Indicate(vtex[2])]
        ])
        self.wait()

        self.play(*[Uncreate(item) for item in [
            circles,
            atex,
            vtex,
            x, v
        ]])

        self.wait()


class SVDdot3(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            include_foreground_plane=False
        )

    def construct(self):
        matrix = MatrixFromSVD(PI/9, [1.2, 2.5], -PI/6)

        x = Vector([1, 0], color=RED)
        v = Vector([-1, -1], color=GOLD)
        self.add_vector(x)
        self.add_vector(v)

        circles = SVD2Transpose(self, matrix, x, v)
        self.wait()

        # Tear down
        self.play(
            *[ShrinkToCenter(item) for item in [
                self.background_plane,
                circles,
            ]] + [
                FadeOut(x),
                FadeOut(v)
            ], run_time=2.5)
        self.wait()


class InvertTranspose(Scene):
    def construct(self):
        self.wait()

        # Write transforms
        xt = MathTex(r"\bar{x} = A x", font_size=75).shift(UP)
        vft = MathTex(r"\bar{v}", "=", r"A^{-1^T}","v", font_size=75).shift(DOWN)
        self.play(DrawBorderThenFill(xt), DrawBorderThenFill(vft))
        self.wait(2)

        # highlight inverse
        self.play(Indicate(VGroup(vft[2][1], vft[2][2])))
        self.wait(10)

        # Input vectors
        x = MathTex("x", font_size=75).shift(UP+LEFT*4.5).align_to(xt, DOWN)
        v = MathTex("v", font_size=75).shift(DOWN+LEFT*4.5).align_to(vft, DOWN)
        self.play(DrawBorderThenFill(x), DrawBorderThenFill(v))
        self.wait()

        # Output vectors
        xbar = MathTex(r"\bar{x}", font_size=75).shift(
            UP+RIGHT*4.5).align_to(xt, DOWN)
        vbar = MathTex(r"\bar{v}", font_size=75).shift(
            DOWN+RIGHT*4.5).align_to(vft, DOWN)
        self.play(DrawBorderThenFill(xbar), DrawBorderThenFill(vbar))
        self.wait()

        # Indicate input vectors
        self.play(Circumscribe(x), Circumscribe(v))
        self.wait()

        # Draw arros and shift and indicate rules
        xarrow = Arrow(x.get_right()+RIGHT,
                       np.multiply(x.get_right(), [-1, 1, 1])+LEFT, color=BLUE)
        varrow = Arrow(v.get_right()+RIGHT,
                       np.multiply(v.get_right(), [-1, 1, 1])+LEFT, color=BLUE)
        self.play(
            xt.animate.shift(UP),
            vft.animate.shift(DOWN),
            Write(xarrow),
            Write(varrow), run_time=2)
        self.play(Circumscribe(xt), Circumscribe(vft))
        self.wait()

        # Indicate output vectors
        self.play(Circumscribe(xbar), Circumscribe(vbar))
        self.wait(4)

        # Switch V arrow
        varrowr = Arrow(varrow.get_right(), varrow.get_left(), color=GREEN)
        self.play(Transform(varrow, varrowr, path_arc=PI/4))
        self.wait()

        # Flash new combinations
        self.play(
            Circumscribe(x),
            Circumscribe(vbar)
        )
        self.wait()
        self.play(
            Circumscribe(xarrow),
            Circumscribe(varrow),
        )
        self.wait()
        self.play(
            Circumscribe(xbar),
            Circumscribe(v)
        )
        self.wait()

        # Solve for v
        self.play(FocusOn(vft))
        vbt = MathTex("A^T",r"\bar{v}","=","v", font_size=75).move_to(vft)
        #vbt = MathTex(r"v", "=", r"A^T", r"\bar{v}", font_size=75).move_to(vft)
        self.play(
            ReplacementTransform(vft[0],vbt[1], path_arc=PI/2),
            ReplacementTransform(vft[1],vbt[2]),
            ReplacementTransform(vft[3],vbt[3]),
            ReplacementTransform(vft[2][0],vbt[0][0]),
            ReplacementTransform(vft[2][3],vbt[0][1]),
            FadeOut(VGroup(vft[2][1], vft[2][2]), shift=LEFT)
        , run_time=2)
        self.wait()

        # Indicate v equation
        self.play(Circumscribe(vbt))
        self.wait(4)

        # highlight transpose
        self.play(Indicate(vbt[0]))
        self.wait(2)

        # headline equation
        headline0 = MathTex(r"v", r"\cdot x = ",
                            r"\bar{v}", r"\cdot", r"\bar{x}", font_size=100)
        headline = MathTex("(", r"A^T", r"\bar{v}", ")", r"\cdot x = ",
                           r"\bar{v}", r"\cdot", r"Ax", font_size=100).to_edge(UP)
        headline0.shift(
            [0, (headline[4].get_center()[1]-headline0[1].get_center()[1]), 0])

        # Write equation and move diagram down
        arrowmap = VGroup(
            x,
            v,
            xarrow,
            varrow,
            xt,
            vbt,
            xbar,
            vbar
        )
        self.play(
            Write(headline0),
            arrowmap.animate.shift(DOWN), run_time=2)
        self.wait(0.75)

        # Transition equation to final form
        self.play(
            ReplacementTransform(headline0[1], headline[4]),
            ReplacementTransform(headline0[2], headline[5]),
            ReplacementTransform(headline0[3], headline[6]),
            FadeOut(headline0[0], shift=UP),
            FadeOut(headline0[4], shift=UP),
            TransformFromCopy(vbt[0], headline[1]),
            TransformFromCopy(vbt[1], headline[2]),
            TransformFromCopy(
                VGroup(xt[0][3], xt[0][4]),
                headline[7]
            ),
            FadeIn(headline[0], shift=UP),
            FadeIn(headline[3], shift=UP), run_time=2)
        self.wait()

        underline = Line(start=headline.get_left()+DOWN*0.75+LEFT*0.25,
                         end=headline.get_right()+DOWN*0.75+RIGHT*0.25, color=BLUE)
        self.play(Write(underline))
        self.wait(2)

        # Fadeout diagram, center equation
        self.play(FadeOut(arrowmap))
        self.play(VGroup(headline, underline).animate.center())
        self.wait(5)

        # Remove bars
        headlineNoBar = MathTex(r"(A^T ", "v", r")\cdot x = ",
                                "v", r"\cdot Ax", font_size=100).move_to(headline)
        self.play(TransformMatchingTex(headline, headlineNoBar))
        self.play(Indicate(headlineNoBar[1]), Indicate(headlineNoBar[3]))
        self.wait()

        # Add bars back
        self.play(TransformMatchingTex(headlineNoBar, headline))
        self.play(Indicate(headline[2]), Indicate(headline[5]))
        self.wait()

        self.wait()


class TransposeTransform(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            include_foreground_plane=False
        )

    def construct(self):        
        matrix = pematrix
        x = Vector(pex, color=TEAL)
        xdash = DashedVMobject(Vector(np.matmul(pematrix,np.array(pex).T), color=TEAL), num_dashes=10)
        # xdash = DashedVMobject(x, num_dashes=10)
        v = Vector(pev, color=PURPLE)

        # draw axes
        self.remove(self.background_plane)
        self.play(FadeIn(self.background_plane), run_time=3)
        self.wait(2)

        # add x vector and dash on top
        self.add_vector(x)
        self.moving_vectors.remove(x)        
        xlab = MathTex("x", font_size=65, color=TEAL).next_to(x.get_tip(), UL).add_background_rectangle()
        self.play(Write(xlab))
        self.wait(2)
        
        # Add equations
        xeqn = MathTex(r"\bar{x}", "= A", "x", font_size=65).to_edge(
            UR).add_background_rectangle()
        xeqn[1][1].set_color(TEAL)
        xeqn[3].set_color(TEAL)
        self.play(Write(xeqn))
        self.wait()

        # First transform        
        xdashstart = DashedVMobject(x, num_dashes=10)
        self.add(xdashstart)
        self.play(ReplacementTransform(xdashstart, xdash), run_time=3)

        # add xbar label
        xbarlab = MathTex(r"\bar{x}", font_size=65).add_background_rectangle().next_to(xdash.get_corner(DL), DL)
        xbarlab[1][1].set_color(TEAL)
        self.play(Write(xbarlab))
        self.wait()

        # Add dashed v vector
        vdash = DashedVMobject(Vector(np.matmul(np.linalg.inv(matrix.transpose()), v.get_end()[0:2]), color=PURPLE), num_dashes=10)
        self.play(GrowFromEdge(vdash,DL))
        v2 = Vector(np.matmul(np.linalg.inv(matrix.transpose()), v.get_end()[0:2]), color=PURPLE)

        # v label
        vbarlab = MathTex(r"\bar{v}", font_size=65).add_background_rectangle().next_to(v2.get_tip(), UR)
        vbarlab[1][1].set_color(PURPLE)
        self.play(Write(vbarlab))
        self.wait()

        # v equations
        veqn0 = MathTex(r"\bar{v}", font_size=65)
        veqn = MathTex("A^T", r"\bar{v}","=v", font_size=65).to_edge(
            UR).shift(DOWN).add_background_rectangle()
        veqn0.move_to([xeqn[3].get_center()[0],veqn[2].get_center()[1],0]).add_background_rectangle()
        veqn0[1][1].set_color(PURPLE)
        veqn[2][1].set_color(PURPLE)
        veqn[3][1].set_color(PURPLE)
        
        # Write just v-bar
        # self.play(Write(veqn0))
        # self.wait()

        # finish v equation
        # self.play(
        #     ReplacementTransform(veqn0[0],veqn[0]),
        #     ReplacementTransform(veqn0[1],veqn[2]),
        #     FadeIn(veqn[1], shift=LEFT),
        #     FadeIn(veqn[3], shift=LEFT)
        # )
        self.play(Write(veqn))
        self.wait()
        
        # add solid v        
        self.play(FadeIn(v2), run_time=0.5)
        self.moving_vectors.append(v2)

        # transform back
        self.moving_mobjects = []
        self.apply_transposed_matrix(matrix)        

        # write label
        vlab = MathTex("v", font_size=65, color=PURPLE).add_background_rectangle().next_to(v2.get_tip(),DR)
        self.play(Write(vlab))
        self.wait()

        # flash solid then dashed
        self.play(
            ApplyWave(x),
            ApplyWave(v),
            ApplyWave(xeqn[3]),
            ApplyWave(veqn[3][1]),
            ApplyWave(xlab),
            ApplyWave(vlab)
        )
        self.wait(0.5)
        self.play(
            ApplyWave(xdash),
            ApplyWave(vdash),
            ApplyWave(xeqn[1]),
            ApplyWave(veqn[2]),
            ApplyWave(xbarlab),
            ApplyWave(vbarlab)
        )
        self.wait(5)



class TransposeDims(Scene):
    def construct(self):
        # headline equation and underline, centered
        headline = MathTex(
            r"(A^T ", r"\bar{v}", r")\cdot x = ", r"\bar{v}", r"\cdot Ax", font_size=100)
        underline = Line(start=headline.get_left()+DOWN*0.75+LEFT*0.25,
                         end=headline.get_right()+DOWN*0.75+RIGHT*0.25, color=BLUE)
        eqn = VGroup(headline, underline).center()
        self.add(eqn)
        self.wait(4)

        # move equation up
        self.play(eqn.animate.to_edge(UP))

        # Matrices
        A = Matrix([["-", "-", "-"], ["-", "-", "-"]])
        A_Tex = MathTex(
            r"A", ":", r"\mathbb{R}^3", r"\rightarrow", r"\mathbb{R}^2").next_to(A, DOWN)
        letter_A = MathTex("A").next_to(A, DOWN)
        AT = Matrix([["-", "-"], ["-", "-"], ["-", "-"]]).shift(RIGHT*2.5)
        letter_AT = MathTex("A^T").next_to(AT, DOWN)
        AT_Tex = MathTex(
            r"A^T", ":", r"\mathbb{R}^2", r"\rightarrow", r"\mathbb{R}^3").next_to(AT, DOWN)
        A_Tex.align_to(AT_Tex, DOWN)

        # write first matrix
        self.play(Write(A), Write(letter_A))
        self.play(ReplacementTransform(letter_A, A_Tex[0]), Write(A_Tex[1]))
        self.play(Write(A_Tex[2]))
        # self.play(Indicate(A_Tex[2]))
        self.play(Write(A_Tex[3]), Write(A_Tex[4]))
        # self.play(Indicate(A_Tex[4]))
        self.wait()

        # Move first matrix to left
        self.play(VGroup(A, A_Tex).animate.shift(LEFT*2.5))

        # Write second matrix
        self.play(Write(AT), Write(letter_AT))
        self.play(TransformMatchingShapes(
            letter_AT, AT_Tex[0]), Write(AT_Tex[1]))
        self.play(Write(AT_Tex[2]))
        # self.play(Indicate(AT_Tex[2]))
        self.play(Write(AT_Tex[3]), Write(AT_Tex[4]))
        # self.play(Indicate(AT_Tex[4]))
        self.wait(9)

        # Not invertible
        text = Tex("Not Invertible").next_to(A_Tex, DOWN)
        self.play(Write(text))
        self.wait(7)

        # Get rid of stuff
        self.play(
            *[Unwrite(item) for item in [
                headline,
                A,
                AT,
                A_Tex,
                AT_Tex,
                text
            ]],
            Uncreate(underline)
        )
        self.wait()


class SVDFormulas(Scene):
    def construct(self):
        # Common indication parameters
        scale_factor = 1.5
        run_time = 2.55

        # Define SVD formula
        formula1 = MathTex(r"A", r"=", r"R_2", r"\Sigma", r"R_1", font_size=60)
        # Define transpose of A formula
        formula2 = MathTex(r"A^{-1^T}", r"=", r"R_2",
                           r"\Sigma^{-1}", r"R_1", font_size=60)
        # Define inverse of A formula
        formula3 = MathTex(r"A^{-1}", r"=", r"R_1^{-1}",
                           r"\Sigma^{-1}", r"R_2^{-1}", font_size=60)
        # Define transpose of inverse of A formula
        formula4 = MathTex(
            r"A^T", r"=", r"R_1^{-1}", r"\Sigma", r"R_2^{-1}", font_size=60)

        # Position formulas
        formula1.shift(1*UP + 3*LEFT)
        formula3.shift(1*UP + 3*RIGHT).align_to(formula1, DOWN)
        AlignBaseline(formula3, formula1)
        formula2.shift(1*DOWN + 3*LEFT)
        formula4.shift(1*DOWN + 3*RIGHT).align_to(formula2, DOWN)
        AlignBaseline(formula4, formula2)

        # Show forward formulas
        self.play(Write(formula1))
        self.wait(2)
        self.play(TransformFromCopy(formula1, formula2), run_time=1.5)
        self.wait(3)

        # Highlight parts of forward and inv-trans
        self.play(
            Indicate(formula1[4], scale_factor=scale_factor),
            Indicate(formula2[4], scale_factor=scale_factor), run_time=run_time)
        self.play(
            Indicate(formula1[3], scale_factor=scale_factor),
            Indicate(formula2[3], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.play(
            Indicate(formula1[2], scale_factor=scale_factor),
            Indicate(formula2[2], scale_factor=scale_factor), run_time=run_time)
        self.wait()

        # Show inverse
        self.play(TransformFromCopy(formula1.copy(), formula3), run_time=1.5)
        self.wait()

        # highlight inverse similarities
        self.play(
            Indicate(formula1[2], scale_factor=scale_factor),
            Indicate(formula3[4], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.play(
            Indicate(formula1[3], scale_factor=scale_factor),
            Indicate(formula3[3], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.play(
            Indicate(formula1[4], scale_factor=scale_factor),
            Indicate(formula3[2], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.wait()

        # show transpose
        ate = VGroup(formula4[0].copy(), formula4[1].copy())
        self.play(Write(ate))
        self.wait(3)
        self.play(ReplacementTransform(
            formula1.copy(), formula4), run_time=1.5)
        self.remove(ate)
        self.wait()

        # highlight transpose similarities
        self.play(
            Indicate(formula1[2], scale_factor=scale_factor),
            Indicate(formula4[4], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.play(
            Indicate(formula1[3], scale_factor=scale_factor),
            Indicate(formula4[3], scale_factor=scale_factor), run_time=run_time)
        self.play(
            Indicate(formula1[4], scale_factor=scale_factor),
            Indicate(formula4[2], scale_factor=scale_factor, color=BLUE), run_time=run_time)
        self.wait(4)

        # Sentence
        text = Tex(
            "If $A$ is not invertible, then $\Sigma$ is not invertible").shift(3*DOWN)
        self.play(Write(text))
        self.wait()

        # Red out inverts
        formula2r = formula2.copy().set_color(RED).set_opacity(0.5)
        formula3r = formula3.copy().set_color(RED).set_opacity(0.5)
        self.play(
            Transform(formula2, formula2r),
            Transform(formula3, formula3r), run_time=2)
        self.wait(7)

        # Remove most things, for symmetric matrix bit
        self.play(
            FadeOut(formula2),
            FadeOut(formula3),
            FadeOut(formula4),
            FadeOut(text)
        )

        # Move to center, transform to r-inv
        self.play(formula1.animate.move_to(UP))
        formula1s = MathTex(
            r"A", r"=", r"R^{-1}_1", r"\Sigma", r"R_1", font_size=60).move_to(UP)
        self.play(Transform(formula1, formula1s), run_time=1.5)
        self.wait()

        # transform to just r's
        formula1s = MathTex(
            r"A", r"=", r"R^{-1}", r"\Sigma", r"R", font_size=60).move_to(UP)
        self.play(
            ReplacementTransform(
                VGroup(formula1[0], formula1[1], formula1[3]),
                VGroup(formula1s[0], formula1s[1], formula1s[3])),
            ReplacementTransform(formula1[2][0], formula1s[2][0]),
            ReplacementTransform(formula1[2][1], formula1s[2][1]),
            ReplacementTransform(formula1[2][2], formula1s[2][2]),
            FadeOut(formula1[2][3]),
            ReplacementTransform(formula1[4], formula1s[4]), run_time=1.5)
        self.wait()

        # Indicate steps
        self.play(Indicate(formula1s[4]), run_time=2)
        self.play(Indicate(formula1s[3]), run_time=2)
        self.play(Indicate(formula1s[2], color=BLUE), run_time=2)
        self.wait()

        # Put transpose back on, move to center
        self.play(FadeIn(formula4.move_to(DOWN), shift=LEFT), run_time=1.5)
        self.wait()

        # get r into transpose
        formula4s = MathTex(
            r"A^T", r"=", r"R^{-1}", r"\Sigma", r"R", font_size=60).move_to(DOWN)
        self.play(
            ReplacementTransform(
                VGroup(formula4[0], formula4[1], formula4[3]),
                VGroup(formula4s[0], formula4s[1], formula4s[3])),
            ReplacementTransform(formula4[2][0], formula4s[2][0]),
            ReplacementTransform(formula4[2][1], formula4s[2][1]),
            ReplacementTransform(formula4[2][2], formula4s[2][2]),
            FadeOut(formula4[2][3]),
            ReplacementTransform(formula4[4], formula4s[4]), run_time=2)
        self.wait()

        # Indicate equality
        self.play(
            Indicate(VGroup(formula1s[2], formula1s[3], formula1s[4])),
            Indicate(VGroup(formula4s[2], formula4s[3], formula4s[4])), run_time=1.5)
        self.wait()

        # Merge formulas
        formula1s2 = MathTex(r"A", r"=", r"R^{-1}", r"\Sigma", r"R", "=",
                             "A^T", font_size=60).move_to(UP).align_to(formula1s, DOWN)
        self.play(
            ReplacementTransform(
                VGroup(formula1s[0], formula1s[1],
                       formula1s[2], formula1s[3], formula1s[4]),
                VGroup(formula1s2[0], formula1s2[1], formula1s2[2], formula1s2[3], formula1s2[4])),
            Write(formula1s2[5]),
            ReplacementTransform(formula4s[0], formula1s2[6]),
            ReplacementTransform(
                VGroup(formula4s[2], formula4s[3], formula4s[4]),
                VGroup(formula1s2[2], formula1s2[3], formula1s2[4])
            ),
            FadeOut(formula4s[1]), run_time=2)
        self.wait(5)

        # Title symmetric matrix
        sym_text = Text("Symmetric Matrix").move_to(UP*2)
        underline = Line(start=sym_text.get_left()+DOWN*0.5+LEFT*0.25,
                         end=sym_text.get_right()+DOWN*0.5+RIGHT*0.25, color=BLUE)
        self.play(Write(sym_text), Create(underline))
        self.wait(5)

        # Quick example
        matrix = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]]).move_to(DOWN)
        self.play(Write(matrix))
        self.wait()
        transpose3(self, matrix)
        self.wait(2)

        # Get rid of stuff
        self.play(
            *[FadeOut(item) for item in self.mobjects], run_time=3)
        self.wait()

      


class MultipleTransforms(Scene):
    def construct(self):
        matrix = [[1,1],[0,1]]
        buff=0.35
        display_phantoms = False

        # axes        
        axis = NumberPlane(
            x_range=[-6,6], y_range=[-6,6],
            x_length=6.6, y_length=6.6
            )        
        
        # bounding box
        box = Square(3.5, color=PURPLE).move_to(axis)
                        
        # mask
        fs = FullScreenRectangle()
        mask = always_redraw(lambda: Difference(fs, box).set_fill(BLACK, 1).set_stroke(opacity=0))

        # basis vectors
        basis = VGroup(
                    Arrow(ORIGIN, LEFT, color=GREEN).put_start_and_end_on(axis.get_origin(), axis.c2p(1,0)),
                    Arrow(ORIGIN, LEFT, color=RED  ).put_start_and_end_on(axis.get_origin(), axis.c2p(0,1)),
                )
        
        graph = VGroup(axis, basis, box).set_z_index(-1)
        graph.shift(LEFT)
        
        self.add(graph, mask)                        
        
        # caption
        cap0 = MathTex(r"A",r"=","R_2", r"\Sigma", "R_1", font_size=65).next_to(box)
        for i in [0, 1, slice(2,None)]:
            self.play(Write(cap0[i]))
        self.wait()

        # Forward transform
        circle = SVDAnimByAxis(self, axis, matrix, basis_vectors=basis,
            added_anims=[
                [Indicate(cap0[4], scale_factor=1.5, run_time=3)],
                [Indicate(cap0[3], scale_factor=1.5, run_time=3)],
                [Indicate(cap0[2], scale_factor=1.5, run_time=3)]
            ]                       
        )        
        self.wait()        

        # shift to corner, and condense equation
        # distance = np.array([-4.375,  1.875, 0.]) - axis.get_center()
        distance = box.copy().to_corner(UL, buff=buff).get_center() - box.get_center()
        cap1 = MathTex(r"A\\",r"=\\","R_2", r"\Sigma", "R_1", tex_environment="gather*", font_size=60).next_to(box.copy().shift(distance))
        self.play(
            *[item.animate.shift(distance) for item in [axis, basis, circle, box]],
            TransformMatchingShapes(cap0, cap1)
        , run_time=2)
        self.wait(10)


        # Move things over to do the inverse.
        # Real plan is to leave first graph in place,
        # But i'll get that in post!
        # distance = np.array([ 2.375,  1.875, 0.]) - np.array([-4.375,  1.875, 0.])
        cap2 = MathTex(r"A^{-1}\\",r"=\\","R_1^{-1}", r"\Sigma^{-1}", "R_2^{-1}", tex_environment="gather*", font_size=60).move_to(cap1).to_edge(RIGHT, buff=buff)
        distance = box.copy().next_to(cap2, LEFT).get_center() - box.get_center()        
        if display_phantoms: self.add(cap1.copy(), box.copy().set_z_index(0))
        self.play(
            *[item.animate.shift(distance) for item in [axis, basis, circle, box]],
            *TransformBuilder(
                cap1, cap2,
                [
                    ([0,0],[0,0]), # A
                    (None, [0,slice(1,None)]), # -1
                    (1,1), # =
                    ([4,0],[2,0]), ([4,1],[2,3]), # r1
                    (None, [2, slice(1, 3)]), #-1 for r1
                    ([3,0], [3,0]), # sigma
                    (None, [3,slice(1,None)]), #-1 for sigma
                    ([2,0], [4,0]), ([2,1], [4,3]), # r2
                    (None, [4, slice(1, 3)]), #-1 for r2
                ]
            )
        , run_time=3)
        self.wait()

        # Inverse transform
        SVDAnimByAxis(self, axis, matrix, circle=circle, basis_vectors=basis,
            invert_s = True, invert_r = True,
            added_anims=[
                [Indicate(cap2[4], color=BLUE, scale_factor=1.5, run_time=3)],
                [Indicate(cap2[3], color=BLUE, scale_factor=1.5, run_time=3)],
                [Indicate(cap2[2], color=BLUE, scale_factor=1.5, run_time=3)]
            ]                       
        )        
        self.wait(10)

        # need this for coordinates later
        phantombox2 = box.copy()

        # copy to bottom left for inverse transpose
        cap3 = MathTex(r"A^{-1^T}\\",r"=\\","R_2", r"\Sigma^{-1}", "R_1", tex_environment="gather*", font_size=60).next_to(box.copy().to_corner(DL, buff=buff))
        distance = box.copy().to_corner(DL, buff=buff).get_center() - box.get_center()        
        if display_phantoms: self.add(cap2.copy(), box.copy().set_z_index(0))
        self.play(
            *[item.animate.shift(distance) for item in [axis, basis, circle, box]],
            *TransformBuilder(
                cap2, cap3,
                [                    
                    ([0,slice(0,3)],[0,slice(0,3)]), # A -1                    
                    (None, [0,3]), # -T for A
                    (1,1), # =
                    ([4,0],[2,0]), ([4,3], [2,1]), #r2
                    ([4,slice(1,3)], None), # -1 from r2
                    (3, 3), # sigma inverse                    
                    ([2,0],[4,0]), ([2,3], [4,1]), #r1
                    ([2,slice(1,3)], None) # -1 from r1
                ]
            )
        , run_time=3)        
        self.wait()

        # Inverse-transpose transform
        SVDAnimByAxis(self, axis, matrix, circle=circle, basis_vectors=basis,
            invert_s = True, invert_r = False,
            added_anims=[
                [Indicate(cap3[4], color=YELLOW, scale_factor=1.5, run_time=3)],
                [Indicate(cap3[3], color=BLUE, scale_factor=1.5, run_time=3)],
                [Indicate(cap3[2], color=YELLOW, scale_factor=1.5, run_time=3)]
            ]                       
        )        
        self.wait(10)

        # move everything to bottom right for transpose
        cap4 = MathTex(r"A^T\\",r"=\\","R_1^{-1}", r"\Sigma", "R_2^{-1}", tex_environment="gather*", font_size=60).next_to(box.copy().move_to([phantombox2.get_center()[0],box.get_center()[1],0]))
        distance = np.array([phantombox2.get_center()[0],box.get_center()[1],0]) - box.get_center()                
        if display_phantoms: self.add(cap3.copy(), box.copy().set_z_index(0))
        self.play(
            *[item.animate.shift(distance) for item in [axis, basis, circle, box]],
            *TransformBuilder(
                cap3, cap4,
                [
                    ([0,0],[0,0]), # A
                    ([0,slice(1,3)], None), #remove -1
                    ([0,3], [0,1]), # T
                    (1,1), # =
                    ([4,0],[2,0]), ([4,1],[2,3]), # r1
                    (None, [2, slice(1, 3)]), #-1 for r1
                    ([3,0], [3,0]), # sigma 
                    ([3,slice(1,3)],None), # sigma -1                   
                    ([2,0], [4,0]), ([2,1], [4,3]), # r2
                    (None, [4, slice(1, 3)]), #-1 for r2
                ]
            )
        , run_time=3)
        self.wait()

        # transpose transform
        SVDAnimByAxis(self, axis, matrix, circle=circle, basis_vectors=basis,
            invert_s = False, invert_r = True,
            added_anims=[
                [Indicate(cap4[4], color=BLUE, scale_factor=1.5, run_time=3)],
                [Indicate(cap4[3], color=YELLOW, scale_factor=1.5, run_time=3)],
                [Indicate(cap4[2], color=BLUE, scale_factor=1.5, run_time=3)]
            ]                       
        )        
        self.wait(10)



# Fills in from previous
class MultipleTransformsUL(Scene):
    def construct(self):
        matrix = [[1,1],[0,1]]
        buff=0.35
        display_phantoms = False

        # axes        
        axis = NumberPlane(
            x_range=[-6,6], y_range=[-6,6],
            x_length=6.6, y_length=6.6
            )        
        
        # bounding box
        box = Square(3.5, color=PURPLE).move_to(axis)
                        
        # mask
        fs = FullScreenRectangle()
        mask = always_redraw(lambda: Difference(fs, box).set_fill(BLACK, 1).set_stroke(opacity=0))

        # basis vectors
        basis = VGroup(
                    Arrow(ORIGIN, LEFT, color=GREEN).put_start_and_end_on(axis.get_origin(), axis.c2p(1,0)),
                    Arrow(ORIGIN, LEFT, color=RED  ).put_start_and_end_on(axis.get_origin(), axis.c2p(0,1)),
                )
        
        graph = VGroup(axis, basis, box).set_z_index(-1)
        
        self.add(graph, mask)                        
        
        # caption        
        distance = box.copy().to_corner(UL, buff=buff).get_center() - box.get_center()        
        for item in [axis, basis, box]: item.shift(distance)
        cap1 = MathTex(r"A\\",r"=\\","R_2", r"\Sigma", "R_1", tex_environment="gather*", font_size=60).next_to(box)
        self.add(cap1)
        self.play(Indicate(box))
        self.wait(5)


        # Forward transform
        circle = SVDAnimByAxis(self, axis, matrix, basis_vectors=basis,
            added_anims=[
                [Indicate(cap1[4], scale_factor=1.5, run_time=3)],
                [Indicate(cap1[3], scale_factor=1.5, run_time=3)],
                [Indicate(cap1[2], scale_factor=1.5, run_time=3)]
            ]                       
        )        
        self.wait(10)        

        


        





class Recap(Scene):
    def construct(self):
        # text and underline
        summary = Text("Summary", font_size=50)
        underline = Line(start=summary.get_left()+DOWN*0.5+LEFT*0.25,
                         end=summary.get_right()+DOWN*0.5+RIGHT*0.25, color=BLUE)

        # Show text
        self.play(Write(summary), Create(underline))        
        self.play(VGroup(summary, underline).animate.to_edge(UP))
        self.wait(2)

        # list items
        points = BulletedList(
            r"Linear transformations do not necessarily preserve \\ the dot product",
            r"Only pure rotations/reflections do \\ (orthogonal matrices, $A^T = A^{-1}$)",
            r"To preserve the dot product, transform by $A$ and $A^{-1^T}$",
            r"Or, pick output vector and transform back using $A^T$",
            r"By SVD, $A=R_2 \Sigma R_1$, $A^{-1^T}=R_2 \Sigma^{-1} R_1$, $A^T = R_1^{-1} \Sigma R_2^{-1}$"
        ).next_to(summary, DOWN).shift(DOWN*0.33)
        for point in points:
            self.play(Write(point))
            self.wait(8)
        self.wait(10)

        # fade out
        self.play(*[
            FadeOut(item) for item in [summary, underline, points]
        ], run_time=3)
        




class Outro(Scene):
    def construct(self):
        # headline equation
        headline = MathTex(
            r"(A^T ", r"\bar{v}", r")\cdot x = ", r"\bar{v}", r"\cdot Ax", font_size=100)
        self.play(FadeIn(headline), run_time=3)
        self.wait(10)

        # remove equation
        self.play(Unwrite(headline))

        # credits
        author = Tex("Created by Sam Levey", font_size=80).shift(UP*2)
        self.play(FadeIn(author, shift=RIGHT), run_time=2)

        # thanks
        thanks = Tex(r"Special thanks to: \\ Josh Perlman \\ Conner Howell", font_size=65)
        self.play(FadeIn(thanks, shift=RIGHT))

        # Banner animation
        banner = ManimBanner()
        banner.scale(0.3)
        banner.to_edge(DOWN)
        banner.shift(RIGHT*2)
        self.play(FadeIn(banner))
        made_with = Tex("Made with ")
        made_with.scale(1.5)
        made_with.next_to(banner, LEFT, buff=1.2)
        made_with.align_to(banner.M, DOWN)
        url = Tex("\\verb|https://manim.community|")
        url.next_to(VGroup(made_with, banner), DOWN, buff=-0.2)
        url.align_to(made_with, LEFT)
        self.play(AnimationGroup(
            AnimationGroup(banner.expand(), Write(made_with)),
            FadeIn(url),
            lag_ratio=0.5
        ))
        self.wait(2)

        # Remove things

        self.play(Unwrite(author), Unwrite(thanks))
        self.play(Uncreate(banner), Unwrite(made_with), Unwrite(url))
        self.wait()


# Don't use the rest of these
"""

class Unique(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = pematrix
        nx = np.array(pex)
        nv = np.array(pev)
        
        x = Vector(nx, color=TEAL)
        v = Vector(nv, color=PURPLE)
        vghost = Vector(nv, color=PURPLE)

        self.add_vector(x)
        self.add_vector(v)


        # First transform
        self.moving_mobjects=[]
        self.apply_matrix(matrix)

        # Dash x and leave it behind
        xghost = DashedVMobject(x, num_dashes = 10)
        self.moving_vectors.remove(x)

        # Dash v (add, then move away from solid)
        self.moving_vectors.remove(v)
        vghost1 = DashedVMobject(Vector(np.dot(matrix,nv), color=PURPLE), num_dashes=10)
        ghost_coords = np.dot(np.transpose(np.linalg.inv(matrix)), nv)
        vghost2 = DashedVMobject(Vector(ghost_coords, color=PURPLE), num_dashes=10)
        ghost_anim = Transform(vghost1, vghost2)
               

        # Get to vbar, leave x behind, fade out solid v
        transform = np.matmul(np.transpose(np.linalg.inv(matrix)), np.linalg.inv(matrix))
        self.moving_mobjects=[]
        self.apply_matrix(transform, added_anims=[FadeTransform(x, xghost), ghost_anim, FadeOut(v)])
    
        """ """some useful calculations
        print(np.dot(nx, nv))
        print(np.dot(matrix, nx))
        print(np.linalg.inv(matrix))
        print(np.transpose(np.linalg.inv(matrix)))
        print(np.dot(np.transpose(np.linalg.inv(matrix)), nv))
        print(np.dot(np.dot(matrix, nx),np.dot(np.transpose(np.linalg.inv(matrix)), nv)))
        """ """
        dp = np.dot(nx, nv)
        xbar = np.dot(matrix,nx)

        # Add green line of other possible vectors
        curve = self.background_plane.plot(lambda x: (dp - xbar[0]*x)/xbar[1], color=GREEN).set_depth(20)
        self.play(FadeOut(self.plane))
        self.play(Create(curve))
        self.wait()

        # Move the vector to other points on the line
        vghost1.save_state()
        horiz = -ghost_coords[0]
        vghost1.target = DashedVMobject(Vector([horiz, (dp - xbar[0]*horiz)/xbar[1]], color=PURPLE), num_dashes=10)
        self.play(Transform(vghost1, vghost1.target), run_time=2)
        vghost1.target = DashedVMobject(Vector([-6*horiz, (dp - xbar[0]*horiz*-6)/xbar[1]], color=PURPLE), num_dashes=10)
        self.play(Transform(vghost1, vghost1.target), run_time=2)
        self.play(Restore(vghost1), run_time=2)
        self.wait(8)

        # Get rid of stuff
        self.play(
            *[Uncreate(item) for item in [self.background_plane, xghost, vghost1, curve]]
        )
        self.wait()





class UniqueDemo(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = pematrix
        nx = np.array(pex)

        x = Vector(nx, color=TEAL)
        self.add_vector(x)
        
        self.moving_mobjects=[]
        self.apply_matrix(matrix)
        self.moving_vectors.remove(x)
        self.wait()
        
        anims = []
        for x in range(-5,6):
            for y in range(-3,4):
                v = Vector([x,y], color=PURPLE)
                anims.append(GrowArrow(v))
                self.moving_vectors.append(v)

        self.play(*anims)
        self.wait(2)

        transform = np.matmul(np.transpose(np.linalg.inv(matrix)), np.linalg.inv(matrix))
        self.moving_mobjects=[]
        self.apply_matrix(transform)
        

        self.wait(2)



class NotInvertibleDemo(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=True
        )

    def construct(self):
        matrix = [[1,0],[0,0]]

        self.moving_mobjects = []
        self.apply_matrix(matrix)

        self.wait(2)

        self.reset_plane(Write)
        self.wait()


    def reset_plane(self, animation):
        self.plane = NumberPlane(**self.foreground_plane_kwargs)
        self.play(animation(self.plane))
        self.add_transformable_mobject(self.plane)
        return self
"""
