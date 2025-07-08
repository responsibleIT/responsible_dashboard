import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkClassesComponent } from './benchmark-classes.component';

describe('BenchmarkClassesComponent', () => {
  let component: BenchmarkClassesComponent;
  let fixture: ComponentFixture<BenchmarkClassesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkClassesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkClassesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
