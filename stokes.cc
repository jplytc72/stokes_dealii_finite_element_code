/* ---------------------------------------------------------------------
 *  *
 *   *
 *    * ---------------------------------------------------------------------
 *
 *     */
#include <deal.II/base/logstream.h>               // Log system during compile/run
#include <deal.II/base/quadrature_lib.h>          // Quadrature Rules
#include <deal.II/base/convergence_table.h>       // Convergence Table
#include <deal.II/base/timer.h>                   // Timer class

#include <deal.II/dofs/dof_handler.h>             // DoF (defrees of freedom) Handler
#include <deal.II/dofs/dof_tools.h>               // Tools for Working with DoFs

#include <deal.II/fe/fe_q.h>                      // Continuous Finite Element Q Basis Function
#include <deal.II/fe/fe_values.h>                 // Values of Finite Elements on Cell
#include <deal.II/fe/fe_system.h>                 // System (vector) of Finite Elements

#include <deal.II/grid/tria.h>                    // Triangulation declaration
#include <deal.II/grid/tria_accessor.h>           // Access the cells of Triangulation
#include <deal.II/grid/tria_iterator.h>           // Iterate over cells of Triangulations
#include <deal.II/grid/grid_generator.h>          // Generate Standard Grids
#include <deal.II/grid/grid_out.h>                // Output Grids

#include <deal.II/lac/affine_constraints.h>       // Constraint Matrix
#include <deal.II/lac/dynamic_sparsity_pattern.h> // Dynamic Sparsity Pattern
#include <deal.II/lac/sparse_matrix.h>            // Sparse Matrix
#include <deal.II/lac/sparse_direct.h>            // UMFPACK

#include <deal.II/numerics/vector_tools.h>        // Interpolate Boundary Values
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>  // C++ Output
#include <fstream>   // C++ Output
#include <cmath>     // C++ Math Functions


using namespace dealii;


template <int dim>
class StokesExactSolution : public Function<dim>
{
  public:
    StokesExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                                 Vector<double>  &value) const;
    Tensor<1,dim> gradient(const Point<dim> &p,
                           const unsigned int component) const;

};

template <int dim>
void
StokesExactSolution<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));

  double x = p[0];
  double y = p[1];

  values(0) = cos(M_PI*x);
  values(1) = y*M_PI*sin(M_PI*x);
  values(2) = x*x*y*y;

}

template <int dim>
Tensor<1,dim>
StokesExactSolution<dim>::gradient(const Point<dim> &p,
                             const unsigned int component) const
{
  //Recall the exact solution is
  // double constant = 40.0;
  // u_1(x,y) = values(0) = cos(M_PI*x);
  // u_2(x,y) = values(1) = y*M_PI*sin(M_PI*x);
  // p(x,y)   = values(2) = x*x*y*y;
  //
  // Note that
  //   div u(x,y) = u_{1,x} + u_{2,y}
  //              = -M_PI*sin(M_PI*x) + M_PI*sin(M_PI*x)
  //              = 0
  //   grad u(x,y) = [grad u_{1} ; grad u_{2} ]  
  //               = [u_{1,x}  u_{1,y}  ; u_{2,x}   u_{2,y}]
  //               = [-M_PI*sin(M_PI*x)   0 ; y*M_PI*M_PI*cos(M_PI*x)   M_PI*sin(M_PI*x)]
  //   grad p(x,y) = <p_x      ,      p_y>
  //               = <2*x*y*y  ,      2*x*x*y>
  double x = p[0];
  double y = p[1];

  Tensor<1,dim> return_value;

  switch(component){
    case 0:  // gradient of 1st component of velocity
      return_value[0] = -M_PI*sin(M_PI*x);      // u_{1,x}
      return_value[1] = 0.;                     // u_{1,y}
      return return_value;
      break;
    case 1:  // gradient of 2nd component of velocity
      return_value[0] = y*M_PI*M_PI*cos(M_PI*x);      // u_{2,x}
      return_value[1] = M_PI*sin(M_PI*x);             // u_{2,y}
      return return_value;
      break;
    case 2:
      return_value[0] = 2*x*y*y;
      return_value[1] = 2*x*x*y;
      return return_value;
      break;
    default:
      Assert(false, ExcNotImplemented());
  }

}

template <int dim>
class StokesProblem
{
  public:
    StokesProblem (const unsigned int degree);
    void run ();

    private:
      void setup_dofs ();
      void refine_grid();
      void assemble_stokes_system ();
      void assemble_navier_stokes_system ();
      void stokes_solve ();
      void navier_stokes_solve ();
      void solution_update ();
      void setup_convergence_table();
      void output_results (const unsigned int cycle);
      void compute_errors (const unsigned int cycle);

      const unsigned int  degree;

      Triangulation<dim>      triangulation;
      FESystem<dim>           fe;
      DoFHandler<dim>         dof_handler;

      SparsityPattern         sparsity_pattern;
      SparseMatrix<double>    system_matrix;

      Vector<double>          solution;
      Vector<double>          delta_solution;
      Vector<double>          system_rhs;

      ConvergenceTable        convergence_table;

      ConstraintMatrix        zero_press_node_constraint;
};

// <<<<<<< Constructor

template<int dim>
StokesProblem<dim>::StokesProblem(const unsigned int degree)
    :
    degree(degree),
    fe(FE_Q<dim>(degree+1), dim,
    FE_Q<dim>(degree), 1), 
    dof_handler(triangulation)
    {}

template<int dim>
void StokesProblem<dim>::refine_grid()
{
  std::cout << "============================================================"
            << std::endl
            << "Globally refining domain..."
            << std::endl
            << "-------------------------------------------------------------"
            << std::endl;

  triangulation.refine_global(1);

  std::cout << "-------------------------------------------------------------"
            << std::endl
            << "Completed Globally refining domain..."
            << std::endl
            << "============================================================"
            << std::endl;
// >>>>>>> master
}
